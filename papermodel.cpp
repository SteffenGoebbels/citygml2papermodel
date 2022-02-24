//
//  Unfolding of textured CityGML buildings to paper models
//  =======================================================
//
//  Created by Steffen Goebbels on 11.02.2022, iPattern Institute, Faculty of Electrical Engineering and Computer Science, Niederrhein University of Applied Sciences, Krefeld, Germany
//  steffen.goebbels@hsnr.de
//
//  The algorithm is described in the paper
//  Steffen Goebbels, Regina Pohle-Froehlich: Automatic Unfolding of CityGML Buildings to Paper Models. Geographies 1 (3), 2021, pp. 333–345, https://doi.org/10.3390/geographies1030018
//
//  You can use and modify this program without any restrictions if you cite this paper. The code comes without any warranty or liability, use at your own risk.
//
//  Prerequisites: OpenCV and libCityGML (tested with version 2.3.0) libraries, LaTeX for PDF creation
//
//  The command line tool unfolds textured CityGML buildings to images that are written to the folder of the input data or to a folder specified by -dest. Also, a LaTeX source file "citymodel.tex" is generated that includes the images. Using the command "pdflatex citymodel", a PDF can be generated in which the unfolded models are correctly scaled with respect to the -scale parameter.
//  The tool processes wall, roof, and ground plane polygons. The assumption is that their vertices are organized counter-clockwise, i.e., in the mathematically positive sense, when seen from outside the building. If this assumption is violated, polygons might overlap in the output. The tool only processes ground plane, wall and roof polygons.
//  Textures are mapped using either an affine or a perspective transform. Whereas texture coordinates outside the interval [0, 1] are supported by periodically repeating (WRAP) as well as repeating and mirroring (MIRROR), a general texture transform like implemented in OpenGL is not applied.
//  Currently, four paper sizes are supported: DIN A0 - A4. If a layout image does not fit onto a single page of paper, the image is divided into parts. Colored bars at the boundaries indicate how the parts fit together.

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <citygml/citygml.h>
#include <citygml/citymodel.h>
#include <citygml/texture.h>
#include <citygml/address.h>
#include <citygml/polygon.h>

//Default values and flags
#define RESOLUTION 15.0  //resolution: 15 Pixel per meter
#define GLUE_TABS true   //flag: false: no glue tabs, true: glue tabs
#define WRITE_IMAGE true
#define SCALE 220        //1:SCALE
#define LATEX_OUTPUT true
#define SHOW_IMAGE false //Show layout images while processing?
#define PAPERSIZE 4

//--------------------------------------------------------
//Structures
//--------------------------------------------------------

//All 3D vertices are stored in a vector separated from polygons and edges.
//Each vertex is stored only once.
//The vector inPolygon contains references to the polygons in which the current vertex is used.
struct vertex_adj_t {
    cv::Point3d p;
    std::vector<int> inPolygon; //the vertex is part of the polygons with these numbers
};

//Role of a polygon
#define GROUND_PLANE 0
#define WALL 1
#define ROOF 2
#define GLUE_TAB_POLYGON 3

//WRAP_MODE
#define MIRROR 1
#define CLAMP 2
#define BORDER 3
#define NONE 4

//Each polygon is stored using the following structure that is iteratively filled by the algorithm
struct texturedPolygon_t {
    int type; //role of the polygon: GROUND_PLANE, WALL, ROOF or GLUE_TAB_POLYGON
    int transformed; //iteration in which the polygon was drawn into a layout image
    std::vector<unsigned int> polygon; //references to separately stored vertices of the exterior polygon, see struct vertex_adj_t.
    std::vector<unsigned int> polygon_simplified; //same as polygon but without intermediate points
    std::vector<std::vector<unsigned int>> interior_polygons; //each polygon can contain interior polygons (holes), their vertices are referenced, see struct vertex_adj_t.
    std::vector<cv::Point2d> coordinates2D; //3D vertices will be transformed to 2D coordinates within the 3D plane on which the planar polygon lies
    std::vector<cv::Point2d> coordinates2D_simplified;
    std::vector<std::vector<cv::Point2d>> interior_coordinates2D; //same as before for interior polygons
    std::vector<cv::Point2d> image_coordinates; //The 2D coordinates with respect to the 3D plane will be mapped to 2D layout coordinates with respect to the image to be printed
    std::vector<cv::Point2d> image_coordinates_simplified;
    std::vector<std::vector<cv::Point2d>> interior_image_coordinates; //same as before for interior polygons
    std::vector<cv::Point2d> texture_coordinates; //This vector has to be either empty or its size has to match the size of the vector polygon_simplified. Texture coordinates of interior polygons are not used.
    std::string texturename; //file name of texture image
    int WrapMode; //default: WRAP, otherwise MIRROR, CLAMP, BORDER
    cv::Point3d normal; //outer normal
    double area_size; //size of the area covered by the polygon
    std::string id; //polygon id from CityGML data
    cv::Point3d centerpoint, direction1, direction2; //the plane of the current polygon
};

//Edge types:
#define NOT_DRAWN 0
#define DRAWN 1
#define MARK_GLUE_TAB 2
//Structure to store fold edges so that they can be differentiated from glue edges
struct fold_edge_t {
    int vertex[2]; //vertices of a fold edge
    int type; //NOT_DRAWN,DRAWN, MARK_GLUE_TAB
};

//Structure to store candidate edges to attach a given polygon to
struct attachTo_t {
    cv::Point2d left, right; //points of the 2D edge on the paper to attach to
    int left_vertex, right_vertex; //corresponding indices of 3D vertices
    int left_index, right_index; //corresponding index positions in the current polygon
    float edge_length; //length of edge
    int numberCommonEdges; // >1 if there are further common edged between the polygons
    int polygonToAttachTo; //number of polygon to attach to
    float weight; //sorting parameter
    std::vector<struct fold_edge_t> foldEdges; //edges without glue tabs
};

//Auxiliary structure for sorting
struct order_t {
    int index;
    float area_size;
};

//Structure for image tiles that are used if an image does not fit onto the paper
struct tile_t {
    cv::Mat image;
    bool empty;
};

//--------------------------------------------------------
//Global variables
//--------------------------------------------------------

//********** for testing purposes
//Ein- und Ausgabedaten



std::string path_textures;
std::string path_output;


//command line parameters
double scale, resolution;
int papersize;
bool output_LaTeX;


int line_thickness=1;
double pixelwidth_x, pixelwidth_y;

//Parsed CityGML structure
std::shared_ptr<const citygml::CityModel> city;

//LaTeX-Datei
FILE *latexfile;

//--------------------------------------
// functions
//--------------------------------------

//Read CityGML model
std::shared_ptr<const citygml::CityModel> readModel(char *filename) {

    std::shared_ptr<const citygml::CityModel> city;
    citygml::ParserParams params;
    
    params.pruneEmptyObjects=true;
    params.keepVertices=true;
    params.optimize=true;
    
    try{
        city = citygml::load( filename, params );
    }catch(...){
        std::cout << "Parser error, check if input filename is correct." << std::endl;
        return NULL;
    }
    return city;
}

//compute normal of polygon, orientation is not specified
bool computeNormal(struct texturedPolygon_t &polygon, std::vector<struct vertex_adj_t>& vertices) {
    
    int k,l;
    
    if(polygon.type == GROUND_PLANE) { //normal of ground plane points downwards
        polygon.normal.x=0.0;
        polygon.normal.y=0.0;
        polygon.normal.z=-1.0;
        polygon.area_size=100000000;
        return false;
    }

    //Select three vertices:
    //Search to vertices with a largest mutual distance. Then add a third vertex such that a determinant becomes largest
    int one=0, two=1, three=2;
    
    double max_distance=0;
    for(k=0; k<(int)polygon.polygon.size()-1; k++) {
        for(l=k+1; l<polygon.polygon.size(); l++) {
            cv::Point3d point = vertices[polygon.polygon[k]].p-vertices[polygon.polygon[l]].p;
            double distance = point.dot(point);
            if(distance > max_distance) {
                max_distance=distance;
                one=k; two=l;
            }
        }
    }
    double max_length_crossproduct=0.0;
    for(k=0;k<polygon.polygon.size();k++) {
        if((k!=one)&&(k!=two)) {
            
            //Compute cross product
            cv::Point3d vec1 = vertices[polygon.polygon[two]].p-vertices[polygon.polygon[one]].p;
            cv::Point3d vec2 = vertices[polygon.polygon[k]].p-vertices[polygon.polygon[one]].p;
            cv::Point3d cross = vec1.cross(vec2);
            
            double lae=cross.dot(cross);
            
            if(lae>max_length_crossproduct) {
                max_length_crossproduct=lae;
                three=k;
            }
        }
    }
    
    if(max_length_crossproduct==0.0) {
        return true; //no normal
    }
     
    cv::Point3d vec1 = vertices[polygon.polygon[two]].p-vertices[polygon.polygon[one]].p;
    cv::Point3d vec2 = vertices[polygon.polygon[three]].p-vertices[polygon.polygon[one]].p;
    cv::Point3d cross = vec1.cross(vec2);
    double lae=cross.dot(cross);
    if(lae>0) {
        polygon.normal=cross/sqrt(lae);
        return(false);
    } else return true; //no valid normal vector
}

//Compute a point and two vectors that span the plane of the polygon
void computePlane(struct texturedPolygon_t &polygon, std::vector<struct vertex_adj_t>& vertices) {
     //Compute center of gravity of vertices and longest edge
    cv::Point3d center;
    cv::Point3d normal=polygon.normal;
    cv::Point3d longestDirection;
    cv::Point3d orthogonalDirection;
    double length=0;
    int anz = -1+(int)polygon.polygon.size();
    for(int i=0; i<anz; i++) {
        cv::Point3d  direction;
        center = center + vertices[polygon.polygon[i]].p;
        direction = vertices[polygon.polygon[i+1]].p - vertices[polygon.polygon[i]].p;
        double l=direction.dot(direction);
        if(l>=length) {
            longestDirection = direction;
            length=l;
        }
    }
    center=center/(double)anz;
    if(length>0) longestDirection=longestDirection/sqrt(length);
    //compute direction orthogonal to normal and to longest direction
    orthogonalDirection=normal.cross(longestDirection);
    length=orthogonalDirection.dot(orthogonalDirection);
    if(length>0) orthogonalDirection=orthogonalDirection/sqrt(length);
    
    polygon.centerpoint=center;
    polygon.direction1=longestDirection;
    polygon.direction2=orthogonalDirection;
}

//compute 2D coordinates in the 3D plane of a polygon
void polygonTo2DPlane(struct texturedPolygon_t &polygon, std::vector<struct vertex_adj_t>& vertices, std::vector<texturedPolygon_t>& allPolygons) {

    for(int k=0; k < polygon.polygon.size(); k++) {
        
        //compute local 2D coordinates
        cv::Point3d v = vertices[polygon.polygon[k]].p-polygon.centerpoint;
        
        //coordinates with respect to the basis:
        
        cv::Point2d point;
        point.x= v.dot(polygon.direction1);
        point.y= v.dot(polygon.direction2);
        
        polygon.coordinates2D.push_back(point);
    }
    for(int k=0; k < polygon.polygon_simplified.size(); k++) {
        
        //compute local 2D coordinates
        cv::Point3d v = vertices[polygon.polygon_simplified[k]].p-polygon.centerpoint;
        
        //coordinates with respect to the basis:
        
        cv::Point2d point;
        point.x= v.dot(polygon.direction1);
        point.y= v.dot(polygon.direction2);
        
        polygon.coordinates2D_simplified.push_back(point);
    }
    
    //Compute area size of polygon
    //If orientation is correct: size has to be positive, otherwise change orientation
    
    double size=0;
    for(int i=0; i<-1+(int)polygon.coordinates2D_simplified.size(); i++) {
        size+=polygon.coordinates2D_simplified[i].x*polygon.coordinates2D_simplified[i+1].y-polygon.coordinates2D_simplified[i].y*polygon.coordinates2D_simplified[i+1].x;
    }
    double factor=1.0;
    if(size<0) {
        factor=-1.0; size*=-1.0;
        for(int i=0; i<polygon.coordinates2D.size(); i++) {
            polygon.coordinates2D[i].y*=-1.0;
        }
        for(int i=0; i<polygon.coordinates2D_simplified.size(); i++) {
            polygon.coordinates2D_simplified[i].y*=-1.0;
        }
    }
    polygon.area_size = size;
    
    for(int i=0; i< polygon.interior_polygons.size(); i++) {
        std::vector<cv::Point2d> innere;
        for(int k=0; k < polygon.interior_polygons[i].size(); k++) {
            
            //compute local 2D coordinates
            cv::Point3d v = vertices[polygon.interior_polygons[i][k]].p-polygon.centerpoint;
            
            //coordinates
            cv::Point2d point;
            point.x= v.dot(polygon.direction1);
            point.y= factor*v.dot(polygon.direction2);
            
            innere.push_back(point);
        }
        polygon.interior_coordinates2D.push_back(innere);
    }
}

//search for a (wall) polygon that has the given (footprint) edge
int searchPolygonForEdge(int left, int right, int type, int &indexLeftBottom, int &indexLeftTop, int &indexRightBottom , int &indexRightTop, std::vector<struct vertex_adj_t>& vertices, std::vector<texturedPolygon_t>& allPolygons) {
    
    for(int i=0; i<vertices[left].inPolygon.size(); i++) {
        for(int j=0; j<vertices[right].inPolygon.size(); j++) {
            if(vertices[left].inPolygon[i]==vertices[right].inPolygon[j]) {
                if(allPolygons[vertices[left].inPolygon[i]].type == type) {
                    //check order: left (links) is followed by right (rechts)
                    for(int k=0; k<-1+(int)allPolygons[vertices[left].inPolygon[i]].polygon.size(); k++) {
                        if(allPolygons[vertices[left].inPolygon[i]].polygon[k]==left) {
                            if(allPolygons[vertices[left].inPolygon[i]].polygon[k+1]==right) {
                                indexLeftBottom=k;
                                indexLeftTop=k-1;
                                indexRightBottom=k+1;
                                indexRightTop=k+2;
                                if(indexLeftTop<0) indexLeftTop=-2+(int)allPolygons[vertices[left].inPolygon[i]].polygon.size();
                                if(indexRightTop>=allPolygons[vertices[left].inPolygon[i]].polygon.size())
                                    indexRightTop=1;
                                return(vertices[left].inPolygon[i]);
                            }
                        }
                    }
                }
            }
        }
    }
    return(-1);
}

//Transformation of 2D coordinates such that match1 is mapped to attach_point1 and match2 is mapped to attach_point2
bool computeImageCoordinates(struct texturedPolygon_t &polygon, cv::Point2d attach_point1, cv::Point2d attach_point2, cv::Point2d match1, cv::Point2d match2, double &minx, double &miny, double &maxx, double &maxy) {
    
    double l=sqrt( (match2.x-match1.x)*(match2.x-match1.x) + (match2.y-match1.y)*(match2.y-match1.y));
    if(l<0.001) {
        //std::cout << "Kante zu kurz." << std::endl;
        return false;
    }
    //scaling factor
    double r= sqrt((attach_point2.x-attach_point1.x)*(attach_point2.x-attach_point1.x)+(attach_point2.y-attach_point1.y)*(attach_point2.y-attach_point1.y))/l;
    
    //perform shift
    cv::Point3d shift;
    shift.x= attach_point1.x - r*match1.x;
    shift.y= attach_point1.y - r*match1.y;
    
    match2.x=r*match2.x+shift.x;
    match2.y=r*match2.y+shift.y;
    //compute rotation angle alpha
    double alpha_attach = atan2(attach_point2.y-attach_point1.y, attach_point2.x - attach_point1.x );
    double alpha_match  = atan2(match2.y-attach_point1.y, match2.x - attach_point1.x );
    double alpha = alpha_attach - alpha_match;
    
    for(int i=0; i<polygon.coordinates2D.size(); i++) {
        cv::Point2d point;
        point.x = r*polygon.coordinates2D[i].x+shift.x - attach_point1.x;
        point.y = r*polygon.coordinates2D[i].y+shift.y - attach_point1.y;
        
        //rotate:
        double x = cos(alpha) * point.x - sin(alpha) * point.y + attach_point1.x;
        double y = sin(alpha) * point.x + cos(alpha) * point.y + attach_point1.y;
        
        point.x = x;
        point.y = y;
        
        if(x<minx) minx=x;
        if(x>maxx) maxx=x;
        if(y<miny) miny=y;
        if(y>maxy) maxy=y;
        
        polygon.image_coordinates.push_back(point);
    }
    for(int i=0; i<polygon.coordinates2D_simplified.size(); i++) {
        cv::Point2d point;
        point.x = r*polygon.coordinates2D_simplified[i].x+shift.x - attach_point1.x;
        point.y = r*polygon.coordinates2D_simplified[i].y+shift.y - attach_point1.y;
        
        //rotate:
        double x = cos(alpha) * point.x - sin(alpha) * point.y + attach_point1.x;
        double y = sin(alpha) * point.x + cos(alpha) * point.y + attach_point1.y;
        
        point.x = x;
        point.y = y;
        
        polygon.image_coordinates_simplified.push_back(point);
    }
    //adjust coordinates of interior polygons
    for(int k=0; k<polygon.interior_coordinates2D.size(); k++) {
        std::vector<cv::Point2d> innere;
        for(int i=0; i<polygon.interior_coordinates2D[k].size(); i++) {
            cv::Point2d point;
            
            point.x = r*polygon.interior_coordinates2D[k][i].x+shift.x - attach_point1.x;
            point.y = r*polygon.interior_coordinates2D[k][i].y+shift.y - attach_point1.y;
            
            //Drehen:
            double x = cos(alpha) * point.x - sin(alpha) * point.y + attach_point1.x;
            double y = sin(alpha) * point.x + cos(alpha) * point.y + attach_point1.y;
            
            point.x = x;
            point.y = y;
            
            innere.push_back(point);
        }
        polygon.interior_image_coordinates.push_back(innere);
    }
    return true;
}

//for debugging: show layout image
void showImage(cv::Mat &image) {
    cv::Mat skaliert;
    
    if(image.cols==0) return;
    if(image.rows==0) return;
    
    //compute scaling
    double factor1=1024.0/(double)image.cols;
    double factor2=640.0/(double)image.rows;
    double factor=1;
    if(factor1<factor2) factor=factor1;
    else factor=factor2;
    cv::resize(image, skaliert, cv::Size(), factor, factor);
    
    cv::imshow("Scaled image",skaliert);
    cv::waitKey(0);
}

bool equal(cv::Point2d first, cv::Point2d second) {
    if(fabs(first.x-second.x)>0.005) return false;
    if(fabs(first.y-second.y)>0.005) return false;
    return true;
}

//check for intersections of polygon (with number polynr) with all other polygons
//polygons have to be drawn without overlaps
//left and right are the image points of the vertex to attach to
bool collision(int polynr, int iteration, std::vector<texturedPolygon_t>& allPolygons, cv::Point2d left, cv::Point2d right) {
    //return 0;
    //iterate through all edges of polygon polynr
    for(int i=0; i< -1+(int)allPolygons[polynr].image_coordinates.size(); i++) {
        cv::Point2d p1, pp1, r1;
        p1=allPolygons[polynr].image_coordinates[i];
        pp1=allPolygons[polynr].image_coordinates[i+1];
        r1=pp1-p1;
        
        //do not check edge to attach to
        if((equal(p1,left)&&equal(pp1,right))||(equal(p1,right)&&equal(pp1,left))) continue;
        
        //double len=sqrt(r1.x*r1.x+r1.y*r1.y);
        //if(len < 1) continue;
        for(int k=0; k< allPolygons.size(); k++) {
            if(k==polynr) continue;
            if(allPolygons[k].transformed!=iteration) continue;
            
            for(int j=0; j<-1+(int)allPolygons[k].image_coordinates.size(); j++) {
                //Compute intersection point between the edges
                cv::Point2d p2, pp2, r2;
                
                p2=allPolygons[k].image_coordinates[j];
                pp2=allPolygons[k].image_coordinates[j+1];
                
                //equal edges are not allowed
                if( (equal(p1,p2) && equal(pp1,pp2)) || (equal(p1,pp2) && equal(pp1,p2)) ) return true;

                r2=pp2-p2;
                //if(r2.x*r2.x+r2.y*r2.y < 1.0) continue;
                //r*r1 - s*r2   = p2-p1
                double d=r1.x*(-r2.y) - r1.y*(-r2.x);
                if(fabs(d)<0.00001) { //parallel
                    
                    //check if straight lines are equal: compute scaled distance to 0 with Hesse representation
                    if(fabs(p1.x*(-r1.y)+p1.y*r1.x - (p2.x*(-r1.y)+p2.y*r1.x))< 0.1){//2*len
                        cv::Point2d richtung1, richtung2;
                        
                        //check if vertices lie on an edge between the other vertices
                        if((p1!=left)&&(p1!=right)) {
                            richtung1=p2-p1;
                            richtung2=pp2-p1;
                            if(richtung1.x*richtung2.x+richtung1.y*richtung2.y <= -0.0) return true; //collision
                        }
                        if(!equal(pp1,left) && !equal(pp1,right)) {
                            richtung1=p2-pp1;
                            richtung2=pp2-pp1;
                            if(richtung1.x*richtung2.x+richtung1.y*richtung2.y <= -0.0) return true; //collision
                        }
                        if(!equal(p2,left) && !equal(p2,right)) {
                            richtung1=p1-p2;
                            richtung2=pp1-p2;
                            if(richtung1.x*richtung2.x+richtung1.y*richtung2.y <= -0.0) return true; //collision
                        }
                        if(!equal(pp2,left) && !equal(pp2,right)) {
                            richtung1=p1-pp2;
                            richtung2=pp1-pp2;
                            if(richtung1.x*richtung2.x+richtung1.y*richtung2.y <= -0.0) return true; //collision
                        }
                    }
                    //edges do not intersect
                    continue;
                }
                double r= ((p2.x-p1.x)*(-r2.y) - (p2.y-p1.y)*(-r2.x))/d;
                double s= (r1.x*(p2.y-p1.y) - r1.y*(p2.x-p1.x))/d;
                //Both vertices at the boundary: no collision
                if((r>0.0001)&&(s>0.0001)&&(r<0.9999)&&(s<0.9999)) {
                    //if(((r>0.1)&&(r<0.9))||((s>0.1)&&(s<0.9)))
                       return true; //collision
                }

                //Check for collisions at vertices
                if((fabs(r)<=0.0001) && !equal(p1,left) && !equal(p1,right) && (s>=0) && (s<=1)) {
                    //intersection at vertex that does not belong to the edge to attach to
                    return true;
                }
                if((fabs(r-1.0)<0.0001) && !equal(pp1,left) && !equal(pp1,right) && (s>=0) && (s<=1)) {
                    //intersection at vertex that does not belong to the edge to attach to
                    return true;
                }
                if((fabs(s)<=0.0001) && !equal(p2,left) && !equal(p2,right) && (r>=0) && (r<=1)) {
                    //intersection at vertex that does not belong to the edge to attach to
                    return true;
                }
                if((fabs(s-1.0)<0.0001) && !equal(pp2,left) && !equal(pp2,right) && (r>=0) && (r<=1)) {
                    //intersection at vertex that does not belong to the edge to attach to
                    return true;
                }
            }
        }
    }
    return false;
}

int cmpInt(const void *a, const void *b) {
    int x = *(int *) a;
    int y = *(int *) b;
    
    return (x-y);
}

//remove empty border regions of an image
cv::Mat removeBorders(cv::Mat& image, bool& empty) {

    int links=-1, rechts=0, oben=0, unten=0;
    empty=false;
    for(int x=0; x<image.cols; x++) {
        for(int y=0; y < image.rows; y++) {
            if(image.at<cv::Vec3b>(y,x)!=cv::Vec3b(255, 255, 255)) {
                links=x;
                x=image.cols;
                break;
            }
        }
    }
    if(links>=0) {
        for(int x=(int)image.cols-1; x>=0; x--) {
            for(int y=0; y < image.rows; y++) {
                if(image.at<cv::Vec3b>(y,x)!=cv::Vec3b(255, 255, 255)) {
                    rechts=x;
                    x=-1;
                    break;
                }
            }
        }
        for(int y=0; y < image.rows; y++) {
            for(int x=0; x<image.cols; x++) {
                if(image.at<cv::Vec3b>(y,x)!=cv::Vec3b(255, 255, 255)) {
                    oben=y;
                    y=image.rows;
                    break;
                }
            }
        }
        for(int y=(int)image.rows-1; y>=0; y--) {
            for(int x=0; x<image.cols; x++) {
                if(image.at<cv::Vec3b>(y,x)!=cv::Vec3b(255, 255, 255)) {
                    unten=y;
                    y=-1;
                    break;
                }
            }
        }
    }
    if(links<0) {
        cv::Rect grid_rect(0,0, 0,0);
        empty=true;
        return image(grid_rect);
    }
    cv::Rect grid_rect(links, oben, rechts-links+1, unten-oben+1);
    return image(grid_rect);
}

//draw 2D layout image
void drawLayoutImages(double &minx, double &miny, double &maxx, double &maxy, int iteration, std::vector<struct fold_edge_t>& foldEdges, std::vector<texturedPolygon_t>& allPolygons) {
    
    static double rowPosition=0;
    static int lastheight=0;
    
    //add a small boundary for glue tabs
    double adjust=ceil(scale/220);
    minx-=2*adjust*resolution;
    maxx+=2*adjust*resolution;
    miny-=2*adjust*resolution;
    maxy+=2*adjust*resolution;
    
    int height = maxy - miny+1;
    int width = maxx - minx +1;
    cv::Mat image(height, width, CV_8UC3, cv::Vec3b(255, 255, 255));
    cv::Mat textureImage;

    std::string lastImage;
    bool empty=true;
    
    for(int i=0; i<allPolygons.size(); i++) {
        if(allPolygons[i].transformed != iteration) continue; //Polygon nicht abgebildet
        if(allPolygons[i].type==GLUE_TAB_POLYGON) continue;
        
        cv::Mat walltexture(height, width, CV_8UC3, cv::Vec3b(255, 255, 255));
        
        empty=false;
        bool correspondingTexture=false;

        int numberCoordinates = (int)allPolygons[i].texture_coordinates.size();
        
        if(numberCoordinates>3) {
            
            if(lastImage!=allPolygons[i].texturename) {
                std::stringstream name;
                std::string buffer;
                name.str(std::string());
                name << path_textures << allPolygons[i].texturename;
                buffer = name.str();
                    
                //std::cout << "Versuche zu lesen: " << buffer << std::endl;
                    
                //buffer = buffer.replace(buffer.length()-4,4,".png");
                textureImage=cv::imread(buffer);
                cv::flip(textureImage,textureImage,0);
                lastImage = allPolygons[i].texturename;
                //showImage(textureImage);
            }
            if((textureImage.rows!=0) && (textureImage.cols!=0)) {
                
                cv::Point2f corner[4]; //4 points in the texture image
                cv::Point2f dest[4];  //4 points in the layout image
                
                //start at the left side and look for further vertices with a maximal value of the minimum of distances to the previously selected vertices
                int index[5]={0,1,2,3,0};

                if(numberCoordinates>4) {
                    
                    for(int k=1; k<-1+(int)allPolygons[i].image_coordinates_simplified.size(); k++ ) {
                        if(allPolygons[i].image_coordinates_simplified[k].x < allPolygons[i].image_coordinates_simplified[index[0]].x)
                            index[0]=k;
                    }
                    
                    for(int kk=1; kk< 3; kk++) {
                        double maxdistance=-1;
                        
                        for(int kkk=0; kkk < -1+(int)allPolygons[i].image_coordinates_simplified.size(); kkk++) {
                            //the index must not be selected before
                            bool used=false;
                            for(int l=0; (l<kk)&& !used; l++) {
                                if(index[l]==kkk) used=true;
                            }
                            if(used) continue;
                            
                            cv::Point2d diff = allPolygons[i].image_coordinates_simplified[index[0]]-allPolygons[i].image_coordinates_simplified[kkk];
                            double mindist=diff.dot(diff);
                            
                            for(int ii=1; ii<kk; ii++) {
                                diff=allPolygons[i].image_coordinates_simplified[index[ii]]-allPolygons[i].image_coordinates_simplified[kkk];
                                double d=diff.dot(diff);
                                if(d<mindist) mindist=d;
                            }
                            
                            if(maxdistance<mindist) {
                                maxdistance=mindist;
                                index[kk]=kkk;
                            }
                        }
                    }
                   
                    //search fourth point that does not lie on the three straight lines through the other points
                    double maxdet=-1;
                 
                    cv::Point2d r[3];
                    r[0]=allPolygons[i].image_coordinates_simplified[index[1]]-allPolygons[i].image_coordinates_simplified[index[0]];
                    r[1]=allPolygons[i].image_coordinates_simplified[index[2]]-allPolygons[i].image_coordinates_simplified[index[0]];
                    r[2]=allPolygons[i].image_coordinates_simplified[index[2]]-allPolygons[i].image_coordinates_simplified[index[1]];
                    
                    for(int kkk=0; kkk < -1+(int)allPolygons[i].image_coordinates_simplified.size(); kkk++) {
                        //the index must not be selected before
                        bool used=false;
                        for(int l=0; (l<3)&& !used; l++) {
                            if(index[l]==kkk) used=true;
                        }
                        if(used) continue;
                        
                        cv::Point2d s[3];
                        s[0]=allPolygons[i].image_coordinates_simplified[kkk]-allPolygons[i].image_coordinates_simplified[index[0]];
                        s[1]=s[0];
                        s[2]=allPolygons[i].image_coordinates_simplified[kkk]-allPolygons[i].image_coordinates_simplified[index[1]];
                        
                        double mindet=-1;
                        for(int j=0;j<3;j++) {
                            double det=fabs(s[j].x*r[j].y-s[j].y*r[j].x);
                            if(mindet==-1) mindet=det;
                            else if(det<mindet) mindet=det;
                        }
                        if(mindet > maxdet) {
                            maxdet=mindet;
                            index[3]=kkk;
                        }
                    }
                    if(maxdet<=0.01) {
                        //std::cout << "Warning: Problem finding fourth point for texture with " << numberCoordinates << " points" << std::endl;
                        numberCoordinates=3; //no additional point found
                    }
                     
                }
                if(numberCoordinates > 4) numberCoordinates = 4;
                else numberCoordinates = 3;
                
                //sort the index
                
                if(numberCoordinates ==4) {
                    qsort(index,4,sizeof(int),cmpInt);
                    index[4]=index[0];
                } else {
                    qsort(index,3,sizeof(int),cmpInt);
                    index[3]=index[0];
                }
                
                //----------------------------------
                //Workaround for degenerated textures that occur in the Berlin city model (Reichstag)
                //If the area size of the texture coordninate polygon is zero, neither
                //affine nor perspective transform can be computed 0> slightly change coordinates
                double size=0.0;
                for(int k=0; k<numberCoordinates; k++) {
                   size+=allPolygons[i].texture_coordinates[index[k]].x*allPolygons[i].texture_coordinates[index[k+1]].y-allPolygons[i].texture_coordinates[index[k]].y*allPolygons[i].texture_coordinates[index[k+1]].x;
                }
                if(fabs(size)<0.000001) {
                    numberCoordinates=3;
                    for(int k=0; k<numberCoordinates; k++) {
                        allPolygons[i].texture_coordinates[index[k]].x+=0.000001*(double)(rand()%100);
                        allPolygons[i].texture_coordinates[index[k]].y+=0.000001*(double)(rand()%100);
                    }
                }
                //-----------------------------------
                
                
                for(int k=0; k<numberCoordinates; k++) {
                    corner[k].x=allPolygons[i].texture_coordinates[index[k]].x;
                    corner[k].y=allPolygons[i].texture_coordinates[index[k]].y;
                    
                    dest[k].x=allPolygons[i].image_coordinates_simplified[index[k]].x-minx;
                    dest[k].y=allPolygons[i].image_coordinates_simplified[index[k]].y-miny;
                }
                
                double texminx=allPolygons[i].texture_coordinates[0].x;
                double texminy=allPolygons[i].texture_coordinates[0].y;
                double texmaxx=texminx;
                double texmaxy=texminy;
                for(int k=1; k<-1+(int)allPolygons[i].texture_coordinates.size(); k++) {
                    if(allPolygons[i].texture_coordinates[k].x < texminx) texminx=allPolygons[i].texture_coordinates[k].x;
                    else if(allPolygons[i].texture_coordinates[k].x > texmaxx) texmaxx=allPolygons[i].texture_coordinates[k].x;
                    
                    if(allPolygons[i].texture_coordinates[k].y < texminy) texminy=allPolygons[i].texture_coordinates[k].y;
                    else if(allPolygons[i].texture_coordinates[k].y > texmaxy) texmaxy=allPolygons[i].texture_coordinates[k].y;
                }
                texminx=floor(texminx);
                texmaxx=ceil(texmaxx);
                texminy=floor(texminy);
                texmaxy=ceil(texmaxy);
                
                cv::Mat pt, at;
                
                if((texminx < 0.0) || (texmaxx > 1.0) || (texminy < 0.0) || (texmaxy > 1.0)) {
                    //periodic repetition of image required because texture coordinates exceed [0, 1]
                    //number of repetitions
                    double nx = texmaxx-texminx;
                    double ny = texmaxy-texminy;
                    if(nx==0) nx=1.0; //saftey if polygon is degenerated
                    if(ny==0) ny=1.0;
                
                    cv::Mat repeatedx, repeatedxy;
                    cv::Mat inputarrayx[(int)nx];
                    cv::Mat inputarrayxy[(int)ny];
                    cv::Mat emptyMatrix(textureImage.rows,textureImage.cols,CV_8UC3,cv::Vec3b(255,255,255));
                    
                    //CLAMP not implemented
                    
                    if(allPolygons[i].WrapMode == MIRROR) {
                        for(int k=0; k<nx; k++) {
                            if(((int)texminx+k)%2) {
                                cv::flip(textureImage, inputarrayx[k],1);
                            } else {
                                inputarrayx[k]=textureImage;
                            }
                        }
                    } else if((allPolygons[i].WrapMode == NONE) || (allPolygons[i].WrapMode == BORDER)  || (allPolygons[i].WrapMode == CLAMP)) {
                        for(int k=0; k<nx; k++) {
                            if(((int)texminx+k)==0) {
                                inputarrayx[k]=textureImage;
                            } else {
                                inputarrayx[k]=emptyMatrix;
                            }
                        }
                    } else {
                        for(int k=0; k<nx; k++) inputarrayx[k]=textureImage;
                    }
                    cv::hconcat(inputarrayx,(int)nx,repeatedx);
                    cv::Mat emptyImagex(repeatedx.rows,repeatedx.cols,CV_8UC3,cv::Vec3b(255,255,255));
                    
                    if(allPolygons[i].WrapMode == MIRROR) {
                        for(int k=0; k<ny; k++) {
                            if(((int)texminy+k)%2) {
                                cv::flip(repeatedx, inputarrayxy[k],0);
                            } else {
                                inputarrayxy[k]=repeatedx;
                            }
                        }
                        std::cout << "MIRROR " << std::endl;
                    } else if((allPolygons[i].WrapMode == NONE) || (allPolygons[i].WrapMode == BORDER)  || (allPolygons[i].WrapMode == CLAMP)) {
                        for(int k=0; k<ny; k++) {
                            if(((int)texminy+k)==0) {
                                inputarrayxy[k]=repeatedx;
                            } else {
                                inputarrayxy[k]=emptyImagex;
                            }
                        }
                        std::cout << "NONE " << std::endl;
                    } else {
                        for(int k=0; k<ny; k++) inputarrayxy[k]=repeatedx;
                    }
                    cv::vconcat(inputarrayxy,(int)ny,repeatedxy);
                    //map texture coordinates to interval [0, 1]
                    for(int k=0; k< numberCoordinates; k++) {
                        corner[k].x=(corner[k].x-texminx)/nx*(double)repeatedxy.cols;
                        corner[k].y=(corner[k].y-texminy)/ny*(double)repeatedxy.rows;
                    }
                    if(numberCoordinates >= 4) {
                        pt=cv::getPerspectiveTransform(corner,dest);
                        cv::warpPerspective(repeatedxy, walltexture, pt, image.size());
                    }
                    else {
                        at=cv::getAffineTransform(corner,dest);
                        cv::warpAffine(repeatedxy, walltexture, at, image.size());
                    }
                } else {
                    for(int k=0; k<numberCoordinates; k++) {
                        corner[k].x=corner[k].x*(double)textureImage.cols;
                        corner[k].y=corner[k].y*(double)textureImage.rows;
                    }
                    if(numberCoordinates >= 4) {
                        pt=cv::getPerspectiveTransform(corner,dest);
                        cv::warpPerspective(textureImage, walltexture, pt, image.size());
                    }
                    else {
                        at=cv::getAffineTransform(corner,dest);
                        cv::warpAffine(textureImage, walltexture, at, image.size());
                    }
                }
                
                correspondingTexture=true;
                //showImage(walltextur);
            }
        }
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Point> contour;
        cv::Point point;
        
        for(int k=0; k<allPolygons[i].image_coordinates_simplified.size(); k++) {
            cv::Point p1;
            p1.x=allPolygons[i].image_coordinates_simplified[k].x-minx;
            p1.y=allPolygons[i].image_coordinates_simplified[k].y-miny;
            contour.push_back(p1);
        }
        contours.push_back(contour);
        
        for(int j=0; j<allPolygons[i].interior_image_coordinates.size(); j++) {
            std::vector<cv::Point> contour;
            for(int k=0; k<allPolygons[i].interior_image_coordinates[j].size(); k++) {
               cv::Point p1;
               p1.x=allPolygons[i].interior_image_coordinates[j][k].x-minx;
               p1.y=allPolygons[i].interior_image_coordinates[j][k].y-miny;
               contour.push_back(p1);
            }
            contours.push_back(contour);
        }
        
        if(correspondingTexture) {
            //create mask
            cv::Mat maske(height, width, CV_8U, cv::Scalar(0));
            cv::drawContours(maske, contours, -1, cv::Scalar(255), CV_FILLED);

            for(int j=0; j<image.rows; j++) {
                for(int k=0; k<image.cols; k++) {
                    if(maske.at<unsigned char>(j, k) == 255) {
                        //copy pixel
                        image.at<cv::Vec3b>(j, k) = walltexture.at<cv::Vec3b>(j, k);
                    }
                    //else {
                    //background
                    //    walltexture.at<cv::Vec3b>(j, k)= cv::Vec3b(255,255,255);
                    //}
                }
            }
        }
        else cv::drawContours(image, contours, -1, cv::Scalar(254,254,254), CV_FILLED);
    }
    //draw border lines onto the textures
    for(int i=0; i<allPolygons.size(); i++) {
        if(allPolygons[i].transformed != iteration) continue; //Polygon nicht abgebildet
        if(allPolygons[i].type==GLUE_TAB_POLYGON) continue;

        //Exterior polygon
        for(int k=0; k<-1+(int)allPolygons[i].image_coordinates.size(); k++) {
            
            //Determine role of the edge
            bool found=false;
            int l=0;
            for(l=0; l<foldEdges.size(); l++) {
                if(((allPolygons[i].polygon[k]==foldEdges[l].vertex[0])&&(allPolygons[i].polygon[k+1]==foldEdges[l].vertex[1]))||((allPolygons[i].polygon[k]==foldEdges[l].vertex[1])&&(allPolygons[i].polygon[k+1]==foldEdges[l].vertex[0]))) {
                    found=true;
                    break;
                }
            }
            if(found) {
                if(foldEdges[l].type==NOT_DRAWN) {
                    //draw a dashed line
                    cv::Point2d d=allPolygons[i].image_coordinates[k+1]-allPolygons[i].image_coordinates[k];
                    double length = sqrt(d.dot(d));
                    if(length>0) {
                        d=d/length;
                        double stepsize=8*line_thickness;
                        int no=length/(2*stepsize);
                        
                        for(int n=0; n<no; n++) {
                            cv::Point p1,p2;
                            p1.x=allPolygons[i].image_coordinates[k].x-minx+(stepsize*2.0*(double)n)*d.x;
                            p1.y=allPolygons[i].image_coordinates[k].y-miny+(stepsize*2.0*(double)n)*d.y;
                            
                            p2.x=allPolygons[i].image_coordinates[k].x-minx+(stepsize*(2.0*(double)n+1.0))*d.x;
                            p2.y=allPolygons[i].image_coordinates[k].y-miny+(stepsize*(2.0*(double)n+1.0))*d.y;
                            cv::line(image,p1,p2,cv::Vec3b(0,0,0),line_thickness,8,0);
                        }
                        
                        cv::Point p1, p2;
                        p1.x=allPolygons[i].image_coordinates[k].x-minx+(stepsize*2.0*(double)no)*d.x;;
                        p1.y=allPolygons[i].image_coordinates[k].y-miny+(stepsize*2.0*(double)no)*d.y;;
                        p2.x=allPolygons[i].image_coordinates[k+1].x-minx;
                        p2.y=allPolygons[i].image_coordinates[k+1].y-miny;
                        cv::line(image,p1,p2,cv::Vec3b(0,0,0),line_thickness,8,0);
                    }
                    foldEdges[l].type=DRAWN;
                } else if(foldEdges[l].type==MARK_GLUE_TAB) {
                    //draw a line
                    cv::Point p1, p2;
                    p1.x=allPolygons[i].image_coordinates[k].x-minx;
                    p1.y=allPolygons[i].image_coordinates[k].y-miny;
                    p2.x=allPolygons[i].image_coordinates[k+1].x-minx;
                    p2.y=allPolygons[i].image_coordinates[k+1].y-miny;
                    cv::line(image,p1,p2,cv::Vec3b(0,0,0),line_thickness,8,0);
                }
            } else {
                //cut edge
                //Draw a line
                cv::Point p1, p2;
                p1.x=allPolygons[i].image_coordinates[k].x-minx;
                p1.y=allPolygons[i].image_coordinates[k].y-miny;
                p2.x=allPolygons[i].image_coordinates[k+1].x-minx;
                p2.y=allPolygons[i].image_coordinates[k+1].y-miny;
                cv::line(image,p1,p2,cv::Vec3b(0,0,0),line_thickness,8,0);
                
                if((allPolygons[i].type!=GROUND_PLANE) && GLUE_TABS) {
                    double dx=allPolygons[i].image_coordinates[k+1].y-allPolygons[i].image_coordinates[k].y;
                    double dy=-(allPolygons[i].image_coordinates[k+1].x-allPolygons[i].image_coordinates[k].x);
                    double length=sqrt(dx*dx+dy*dy);
                    if(length > resolution) {
                        dx/=length; dy/=length;
                        double adjust=ceil(scale/220); 
                        
                        //iteratively try smaller angles
                        for(double factor=1; factor<10; factor+=1.0) {
                            
                            double stepsize=(adjust*(double)resolution)/sqrt(factor);
                            
                            texturedPolygon_t testPolygon;
                            testPolygon.image_coordinates.push_back(allPolygons[i].image_coordinates[k+1]);
                            cv::Point2d p3, p4;
                            p3.x=allPolygons[i].image_coordinates[k+1].x+dx*stepsize;
                            p3.y=allPolygons[i].image_coordinates[k+1].y+dy*stepsize;
                            
                            p4.x=allPolygons[i].image_coordinates[k].x+dx*stepsize;
                            p4.y=allPolygons[i].image_coordinates[k].y+dy*stepsize;
                            
                            if(length<2*factor*adjust*(double)resolution)
                         
                                testPolygon.image_coordinates.push_back(0.5*(p3+p4));
                            else {
                                testPolygon.image_coordinates.push_back(p3+(p4-p3)*(adjust*(double)resolution*factor/length));
                                testPolygon.image_coordinates.push_back(p4-(p4-p3)*(adjust*(double)resolution*factor/length));
                            }
                            testPolygon.image_coordinates.push_back(allPolygons[i].image_coordinates[k]);
                            testPolygon.type=GLUE_TAB_POLYGON;
                            testPolygon.transformed=iteration;
                            
                            int nr=(int)allPolygons.size();
                            allPolygons.push_back(testPolygon);
                            if(!collision(nr, iteration, allPolygons, allPolygons[i].image_coordinates[k+1], allPolygons[i].image_coordinates[k])) {
                                
                                //check background color to avaoid overlaps in error situations
                   
                                int treffer=0;
                                for(int kk=1; kk<-1+(int)testPolygon.image_coordinates.size(); kk++) {
                                    cv::Point p1;
                                    p1.x=testPolygon.image_coordinates[kk].x-minx;
                                    p1.y=testPolygon.image_coordinates[kk].y-miny;
                                    if(image.at<cv::Vec3b>(p1.y, p1.x) != cv::Vec3b(255,255,255)) treffer=1;
                                }
                                
                                if(treffer==0) {
                                    //draw glue tabs
                                    for(int kk=0; kk<-1+(int)testPolygon.image_coordinates.size(); kk++) {
                                        cv::Point p1, p2;
                                        p1.x=testPolygon.image_coordinates[kk].x-minx;
                                        p1.y=testPolygon.image_coordinates[kk].y-miny;
                                        p2.x=testPolygon.image_coordinates[kk+1].x-minx;
                                        p2.y=testPolygon.image_coordinates[kk+1].y-miny;
                                        cv::line(image,p1,p2,cv::Vec3b(0,0,0),line_thickness,8,0);
                                    }
                                
                                    //Add a fold edge so that the opposite edge does not get a fold edge, too
                                    struct fold_edge_t foldEdge;
                                    foldEdge.vertex[0]=allPolygons[i].polygon[k];
                                    foldEdge.vertex[1]=allPolygons[i].polygon[k+1];
                                    foldEdge.type=MARK_GLUE_TAB;
                                    foldEdges.push_back(foldEdge);
                                    
                                    //The edge stays in the list of polygons so that there will be no future overlap
                                    break;
                                }
                                allPolygons.pop_back();
                                    
                            } else allPolygons.pop_back();
                        }
                    }
                
                }
            }
        }
        //interior polygons
        for(int j=0; j<allPolygons[i].interior_image_coordinates.size(); j++) {
            for(int k=0; k<-1+(int)allPolygons[i].interior_image_coordinates[j].size(); k++) {
                cv::Point p1, p2;
                p1.x=allPolygons[i].interior_image_coordinates[j][k].x-minx;
                p1.y=allPolygons[i].interior_image_coordinates[j][k].y-miny;
                p2.x=allPolygons[i].interior_image_coordinates[j][k+1].x-minx;
                p2.y=allPolygons[i].interior_image_coordinates[j][k+1].y-miny;
                cv::line(image,p1,p2,cv::Vec3b(0,0,0),line_thickness,8,0);
            }
        }
        //showImage(image);
    }
    
    if(!empty) {
        cv::flip(image,image,0);
        if(SHOW_IMAGE) showImage(image);
        
        if(output_LaTeX) {
            //If broader then wide and to broad for the paper size: rotate
            if(image.cols < image.rows) { cv::transpose(image, image);  cv::flip(image,image,0); }
            
            //Width in cm
            double px = 100.0*pixelwidth_x/(scale*resolution);
            
            int cols=ceil((float)image.cols/pixelwidth_x);
            int rows=ceil((float)image.rows/pixelwidth_y);
            
            struct tile_t raster[rows][cols];
            
            int count=0;
            int yy=0;
            for (int y = 0; y < image.rows; y += pixelwidth_y) {
                int xx=0;
                for (int x = 0; x < image.cols; x += pixelwidth_x) {
                    
                    int rechts=pixelwidth_x;
                    if(x+rechts > image.cols) rechts=-1-x+(int)image.cols;
                    int unten=pixelwidth_y;
                    if(y+unten > image.rows) unten=-1-y+(int)image.rows;
                    
                    cv::Rect grid_rect(x, y, rechts, unten);
                    
                    bool empty=false;
                    cv::Mat ausschnitt = image(grid_rect);
                    cv::Mat beschnitten = removeBorders(ausschnitt, empty);
                    
                    raster[yy][xx].image =  removeBorders(ausschnitt, empty);
                    raster[yy][xx].empty = empty;
                    xx++;
                }
                yy++;
            }
            //pixel /resolution*100.0/scale = 0.2 cm
            //=> pixel = 0.2 * resolution * scale /100
            int pixel = 0.2 * resolution * scale /100 + 1;
            
            //add hints how to connect cutted image parts
            for(int yy=0; yy < rows; yy++) {
                for(int xx=0; xx < cols; xx++) {
                    if(raster[yy][xx].empty) continue;
                    //upper connection
                    if(yy>0) {
                        if(!raster[yy-1][xx].empty) {
                            cv::Mat bar(pixel, raster[yy][xx].image.cols, CV_8UC3, cv::Vec3b(xx*53, yy*53, iteration*25));
                            
                            cv::Mat inputarray[2]={ bar,raster[yy][xx].image };
                            cv::vconcat(inputarray, 2, raster[yy][xx].image);
                        }
                    }
                    //lower connection
                    if(yy<rows-1) {
                        if(!raster[yy+1][xx].empty) {
                            cv::Mat bar(pixel, raster[yy][xx].image.cols, CV_8UC3, cv::Vec3b(xx*53, (yy+1)*53, iteration*25));
                            
                            cv::Mat inputarray[2]={ raster[yy][xx].image, bar };
                            cv::vconcat(inputarray, 2, raster[yy][xx].image);
                        }
                    }
                    //left connection
                    if(xx>0) {
                        if(!raster[yy][xx-1].empty) {
                            cv::Mat bar(raster[yy][xx].image.rows, pixel, CV_8UC3, cv::Vec3b(iteration*25, xx*89, yy*89));
                                                       
                            cv::Mat inputarray[2]={ bar,raster[yy][xx].image };
                            cv::hconcat(inputarray, 2, raster[yy][xx].image);
                        }
                    }
                    //right connection
                    if(xx<cols-1) {
                        if(!raster[yy][xx+1].empty) {
                            cv::Mat bar(raster[yy][xx].image.rows, pixel, CV_8UC3, cv::Vec3b(iteration*25, (xx+1)*89, yy*89));
                                                       
                            cv::Mat inputarray[2]={ raster[yy][xx].image, bar };
                            cv::hconcat(inputarray, 2, raster[yy][xx].image);
                        }
                    }
                }
            }
            
            for(int yy=0; yy < rows; yy++) {
                for(int xx=0; xx < cols; xx++) {
                    if(raster[yy][xx].empty) continue; //empty regions are omitted
                    
                    std::stringstream name;
                    std::string buffer;
                    name.str(std::string());
                    name << path_output << allPolygons[0].id << "-" << count << "-" << iteration << ".png";
                    buffer = name.str();
                    if(WRITE_IMAGE) {
                        if((lastheight>0)&&(raster[yy][xx].image.cols < lastheight)) {
                            cv::transpose(raster[yy][xx].image, raster[yy][xx].image);  cv::flip(raster[yy][xx].image,raster[yy][xx].image,0);
                        }
                        cv::imwrite(buffer,raster[yy][xx].image);
                    }
                    
                    std::stringstream name2;
                    name2.str(std::string());
                    name2 << "./" << allPolygons[0].id << "-" << count << "-" << iteration << ".png";
                    buffer = name2.str();
                    
                    double breite=(((double)raster[yy][xx].image.cols)/resolution)*100.0/scale; //in cm
                    
                    if(lastheight==0) lastheight=raster[yy][xx].image.rows;
                    
                    if(breite>0) {
                        if((iteration>1)||(count>0)) {
                            if(rowPosition+breite>px) {
                                rowPosition=breite+0.2;
                                lastheight=lastheight=raster[yy][xx].image.rows;
                                fprintf(latexfile,"\\\\\n");
                            } else rowPosition+=(breite+0.2);
                        } else {
                            rowPosition=breite+0.2;
                            lastheight=raster[yy][xx].image.rows;
                        }
                        fprintf(latexfile, "\\includegraphics[width=%ftruecm]{%s} ",breite,buffer.c_str());
                    }
                    
                    count++;
                }
            }
        } else {
            std::stringstream name;
            std::string buffer;
            name.str(std::string());
            name << path_output << allPolygons[0].id << "-" << iteration << ".png";
            buffer = name.str();
            if(WRITE_IMAGE)
                cv::imwrite(buffer,image);            
        }
    }
}

bool compareAttachmentEdges(const struct attachTo_t& x, const struct attachTo_t& y) {
    return x.edge_length*x.weight > y.edge_length*y.weight;
}

bool compareAll(const struct order_t& x, const struct order_t& y) {
    return x.area_size > y.area_size;
}

//We first process a set of walls that are connected with the ground plane.
//Afterwards, further polygons have to be attached via the following function.

void addFurtherPolygons(double &minx, double &miny, double &maxx, double &maxy, int iteration, std::vector<int> &rest, std::vector<struct fold_edge_t> &foldEdges, std::vector<struct vertex_adj_t>& vertices, std::vector<texturedPolygon_t>& allPolygons) {
    
    bool changed=true;
    
    std::vector<struct order_t> order;
    for(int i=0; i< allPolygons.size(); i++) {
        if((allPolygons[i].transformed==0) && (allPolygons[i].type!=GLUE_TAB_POLYGON)){
            struct order_t entry;
            entry.index=i;
            entry.area_size=allPolygons[i].area_size;
            order.push_back(entry);
        }
    }
    if(order.size()==0) return;
    
    //Sort in descending order with respect to area size
    std::sort(order.begin(),order.end(), compareAll);
    
    while(changed) {
        changed=false;
        rest.clear();
        for(int index=0; index<order.size(); index++) {
            int i=order[index].index;
            
            if(allPolygons[i].transformed!=0) {
                continue;
            }
            rest.push_back(i);
            //Look for all edges that are suitable to attach to
            std::vector<struct attachTo_t> candidates;
            for(int j=0; j<-1+(int)allPolygons[i].polygon.size(); j++) {
                
                struct attachTo_t candidate;
                candidate.left_vertex = allPolygons[i].polygon[j];
                candidate.right_vertex = allPolygons[i].polygon[j+1];
                candidate.left_index = j;
                candidate.right_index = j+1;
               
                for(int k=0; (k<vertices[candidate.left_vertex].inPolygon.size()); k++) {
                    if(allPolygons[vertices[candidate.left_vertex].inPolygon[k]].type==GLUE_TAB_POLYGON) continue;
                    if(allPolygons[vertices[candidate.left_vertex].inPolygon[k]].transformed == iteration) {
                        bool found=false;

                        //Search the same polygon at the second vertex
                        
                        for(int kk=0; (kk<vertices[candidate.right_vertex].inPolygon.size()) && !found; kk++) {
                            if( vertices[candidate.right_vertex].inPolygon[kk] == vertices[candidate.left_vertex].inPolygon[k] ) {
                                candidate.polygonToAttachTo = vertices[candidate.right_vertex].inPolygon[kk];
                                //Check, if an edge has been found, search image coordinates (edge is oriented from from left to right)
                                
                                for(int jj=0; jj < -1+(int)allPolygons[candidate.polygonToAttachTo].polygon.size(); jj++) {
                                    if(allPolygons[candidate.polygonToAttachTo].polygon[jj]==candidate.right_vertex) {
                                        if(allPolygons[candidate.polygonToAttachTo].polygon[jj+1]==candidate.left_vertex) {
                                            candidate.left=allPolygons[candidate.polygonToAttachTo].image_coordinates[jj+1];
                                            candidate.right=allPolygons[candidate.polygonToAttachTo].image_coordinates[jj];
                                            candidate.numberCommonEdges=1;
                                            struct fold_edge_t foldEdge;
                                            foldEdge.vertex[0]=candidate.right_vertex;
                                            foldEdge.vertex[1]=candidate.left_vertex;
                                            foldEdge.type=NOT_DRAWN;
                                            candidate.foldEdges.push_back(foldEdge);
                                            candidate.edge_length=sqrt( (candidate.left.x-candidate.right.x)*(candidate.left.x-candidate.right.x) + (candidate.left.y-candidate.right.y)*(candidate.left.y-candidate.right.y));
                                            candidates.push_back(candidate);
                                            
                                            found=true;
                                            break;
                                        } else {
                                            found=true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //multiple edges of the same polygon?
            for(int j=0; j<-1+(int)candidates.size(); j++) {
                for(int k=j+1; k<candidates.size(); k++) {
                    if(candidates[j].polygonToAttachTo == candidates[k].polygonToAttachTo) {
                        candidates[j].numberCommonEdges++;
                        candidates[k].numberCommonEdges++;
                        
                        struct fold_edge_t foldEdge;
                        foldEdge.vertex[0]=candidates[k].left_vertex;
                        foldEdge.vertex[1]=candidates[k].right_vertex;
                        foldEdge.type=NOT_DRAWN;
                        candidates[j].foldEdges.push_back(foldEdge);
                        foldEdge.vertex[0]=candidates[j].left_vertex;
                        foldEdge.vertex[1]=candidates[j].right_vertex;
                        foldEdge.type=NOT_DRAWN;
                        candidates[k].foldEdges.push_back(foldEdge);
                    }
                }
            }
            //Determine weights for attachment candidates
            for(int j=0; j<(int)candidates.size(); j++) {
                if(allPolygons[candidates[j].polygonToAttachTo].type==WALL) candidates[j].weight=10;
                if(allPolygons[candidates[j].polygonToAttachTo].type==GROUND_PLANE) candidates[j].weight=10;
                if(candidates[j].numberCommonEdges>1) candidates[j].weight/=100.0;
            }
            //Sort canditate edges (for attaching to) in descending order with respect to length
            std::sort(candidates.begin(),candidates.end(), compareAttachmentEdges);
            
            //Try layouts in computed order
            for(int j=0; j<candidates.size(); j++) {
                
                double merke_minx=minx;
                double merke_maxx=maxx;
                double merke_miny=miny;
                double merke_maxy=maxy;
                
                bool erg=computeImageCoordinates(allPolygons[i], candidates[j].left, candidates[j].right, allPolygons[i].coordinates2D[candidates[j].left_index], allPolygons[i].coordinates2D[candidates[j].right_index], minx, miny, maxx, maxy);
                
                //check if layouts fits on page
                //add space for glue tabs
                double width = pixelwidth_x  - 4.0*ceil(scale/220)*resolution;
                double height = pixelwidth_y - 4.0*ceil(scale/220)*resolution;
                double deltax=maxx-minx;
                double deltay=maxy-miny;
                if(deltax<deltay) {
                    double merke=deltax; deltax=deltay; deltay=merke;
                }
                if( (deltax>width ) || (deltay > height )) {
                    erg=false;
                }
                if(erg && !collision(i, iteration, allPolygons, candidates[j].left, candidates[j].right)) {
                    allPolygons[i].transformed=iteration;
                    changed=true;
                    rest.pop_back();
        
                    for(int l=0; l<candidates[j].foldEdges.size(); l++) {
                        foldEdges.push_back(candidates[j].foldEdges[l]);
                    }
                               
                    break;
                } else {
                    minx=merke_minx; maxx=merke_maxx; miny=merke_miny; maxy=merke_maxy;
                    allPolygons[i].image_coordinates.clear();
                    allPolygons[i].image_coordinates_simplified.clear();
                    allPolygons[i].interior_image_coordinates.clear();
                    allPolygons[i].transformed=0;
                }
                
            }
        }
    }
}

void addRemainingPolygons(double &minx, double &miny, double &maxx, double &maxy, std::vector<int> &rest, int iteration, std::vector<struct fold_edge_t>& foldEdges, std::vector<struct vertex_adj_t>& vertices, std::vector<texturedPolygon_t>& allPolygons) {

    //rest contains indices of missing polygons in descending order
    //the longest edge of the first (largest) polygon ist the start line
    int maxlength=-1;
    cv::Point2d left, right;
    for(int i=0; i<-1+(int)allPolygons[rest[0]].coordinates2D.size(); i++) {
        double l=(allPolygons[rest[0]].coordinates2D[i+1].x-allPolygons[rest[0]].coordinates2D[i].x)*(allPolygons[rest[0]].coordinates2D[i+1].x-allPolygons[rest[0]].coordinates2D[i].x)+(allPolygons[rest[0]].coordinates2D[i+1].y-allPolygons[rest[0]].coordinates2D[i].y)*(allPolygons[rest[0]].coordinates2D[i+1].y-allPolygons[rest[0]].coordinates2D[i].y);
        if(l>maxlength) {
            left=allPolygons[rest[0]].coordinates2D[i]; right=allPolygons[rest[0]].coordinates2D[i+1];
            maxlength=l;
        }
    }
        
    cv::Point2d attach_point1;
    attach_point1.x=attach_point1.y=0;
    cv::Point2d attach_point2;
    attach_point2.x=sqrt(maxlength)*resolution; attach_point2.y=0;
        
    computeImageCoordinates(allPolygons[rest[0]], attach_point1, attach_point2, left, right, minx, miny, maxx, maxy);
        
    allPolygons[rest[0]].transformed=iteration;
       
    //add further polygons
    if(rest.size()>1)
       addFurtherPolygons(minx, miny, maxx, maxy, iteration, rest, foldEdges, vertices, allPolygons);
    else rest.clear();
    
}

void generateLayout(std::vector<struct fold_edge_t>& foldEdges, std::vector<struct vertex_adj_t>& vertices, std::vector<texturedPolygon_t>& allPolygons) {
    if(allPolygons.size()==0) return;
    
    //largest and smallest coordinates of the image to be created:
    double minx=0, miny=0, maxx=0, maxy=0;
    //list of remaining polygons
    std::vector<int> rest;
    
    //Iterate through the edges of the ground plane in reverse order and look for corresponding wall polygons.
    //This polygons are attached to each other in this order.
    
    if(allPolygons[0].type==GROUND_PLANE) {
        
        cv::Point2d attach_point1;
        attach_point1.x=attach_point1.y=0;
        cv::Point2d attach_point2;
        attach_point2.x=attach_point2.y=0;
        bool first=true;
        for(int i=-1+(int)allPolygons[0].polygon.size(); i>0; i--) {
            //Lower vertices of a corresponding wall
            int k=i-1;
            int left=allPolygons[0].polygon[i];
            int right=allPolygons[0].polygon[k];
            int indexLeftDown=0, indexLeftTop=0, indexRightDown=0, indexRightTop=0;
            //find a wall with this edge
            int wallindex=-1;
            // our own models might have intermediate points on the groundplane that have to be ignored:
            while((wallindex==-1)&&(right!=left)) {
                wallindex=searchPolygonForEdge(left,right,WALL,indexLeftDown, indexLeftTop, indexRightDown, indexRightTop, vertices, allPolygons);
                if(wallindex==-1) {
                    k--;
                    if(k<0) k=-2+(int)allPolygons[0].polygon.size();
                    if(k==i) break;
                    right=allPolygons[0].polygon[k];
                }
            }
            
            if(wallindex<0) {
                if(!first) break; //no fitting wall
                continue;
            }
            
            if(k<i) i=k+1; //ignore intermediate vertices
            else i=0;
            
            //only process a wall if left and right edges are vertical
            cv::Point2d left_b, left_t, right_b, right_t;
            left_b.x=vertices[left].p.x; left_b.y=vertices[left].p.y;
            left_t.x=vertices[allPolygons[wallindex].polygon[indexLeftTop]].p.x;
            left_t.y=vertices[allPolygons[wallindex].polygon[indexLeftTop]].p.y;
            right_b.x=vertices[right].p.x; right_b.y=vertices[right].p.y;
            right_t.x=vertices[allPolygons[wallindex].polygon[indexRightTop]].p.x;
            right_t.y=vertices[allPolygons[wallindex].polygon[indexRightTop]].p.y;
            
            if(!equal(left_b,left_t) || !equal(right_b,right_t))
                break;
            
            //check for collision, necessary if walls are not rectangular
            double l=sqrt( (vertices[right].p.x-vertices[left].p.x)*(vertices[right].p.x-vertices[left].p.x) + (vertices[right].p.y-vertices[left].p.y)*(vertices[right].p.y-vertices[left].p.y));
            attach_point2.x=attach_point1.x+l*resolution;
            double merke_minx=minx;
            double merke_maxx=maxx;
            double merke_miny=miny;
            double merke_maxy=maxy;
            
            bool erg = computeImageCoordinates(allPolygons[wallindex], attach_point1, attach_point2, allPolygons[wallindex].coordinates2D[indexLeftDown], allPolygons[wallindex].coordinates2D[indexRightDown], minx, miny, maxx, maxy);
            
            //check if layouts fits on page
            //add space for glue tabs
            double width  = pixelwidth_x - 4.0*ceil(scale/220)*resolution;
            double height = pixelwidth_y - 4.0*ceil(scale/220)*resolution;
            if(!first && ((((merke_maxx-merke_minx) <=  width)&&( (maxx-minx) > width )) ||
                          (((merke_maxy-merke_miny) <= height)&&( (maxy-miny) > height )))) {
                erg=false;
            }
                
            if(erg && !collision(wallindex, 1, allPolygons, allPolygons[wallindex].image_coordinates[indexLeftTop], allPolygons[wallindex].image_coordinates[indexLeftDown])) {
                    
                allPolygons[wallindex].transformed=1;
                if(first==false) {
                    struct fold_edge_t foldEdge;
                    foldEdge.vertex[0]=allPolygons[wallindex].polygon[indexLeftTop];
                    foldEdge.vertex[1]=allPolygons[wallindex].polygon[indexLeftDown];
                    foldEdge.type=NOT_DRAWN;
                    foldEdges.push_back(foldEdge);
                } else first=false;
                
                attach_point1.x=attach_point2.x;
            } else {
                allPolygons[wallindex].image_coordinates.clear();
                allPolygons[wallindex].image_coordinates_simplified.clear();
                allPolygons[wallindex].interior_image_coordinates.clear();
                allPolygons[wallindex].transformed=0;
                minx=merke_minx; maxx=merke_maxx; miny=merke_miny; maxy=merke_maxy;
                break; //do not go on after a collision
            }
        }
    }
    
    addFurtherPolygons(minx, miny, maxx, maxy, 1, rest, foldEdges, vertices, allPolygons);
    
    //Generate output images
    drawLayoutImages(minx, miny, maxx, maxy, 1, foldEdges, allPolygons);
    
    int iteration=2;
    while(rest.size()>0) {
        double lminx=0, lminy=0, lmaxx=0, lmaxy=0;
        addRemainingPolygons(lminx, lminy, lmaxx, lmaxy, rest, iteration, foldEdges, vertices, allPolygons);
        drawLayoutImages(lminx, lminy, lmaxx, lmaxy, iteration, foldEdges, allPolygons);
        iteration++;
    }
}

//check if three vertices lie on a straight line
bool lieOnStraightLine(int v1, int v2, int v3,std::vector<struct vertex_adj_t>&  vertices) {
    
    cv::Point3d d = vertices[v3].p-vertices[v1].p;

    
    //check if there is a solution for
    // vertices[v1]+ r*d = vertices[v2]
    double r=0;
    if(fabs(d.x)>=0.01)
        r=(vertices[v2].p.x-vertices[v1].p.x)/d.x;
    else if(fabs(d.y)>=0.01)
        r=(vertices[v2].p.y-vertices[v1].p.y)/d.y;
    else if(fabs(d.z)>=0.01)
        r=(vertices[v2].p.z-vertices[v1].p.z)/d.z;
    else return true;
    
    if((fabs(vertices[v1].p.x - vertices[v2].p.x + r*d.x)<0.01) && (fabs(vertices[v1].p.y - vertices[v2].p.y + r*d.y)<0.01) && (fabs(vertices[v1].p.z - vertices[v2].p.z + r*d.z)<0.01)) return true;
    
    return false;
}

struct allEdges_t { int l, r; int polynr; };

bool compareEdges(const struct allEdges_t& x, const struct allEdges_t& y) {
    return x.l < y.l;
}

void processBuilding(const citygml::CityObject& building, std::vector<struct vertex_adj_t>& vertices, std::vector<texturedPolygon_t>& allPolygons) {
   
    std::vector<struct allEdges_t> allEdges;
    //iteratively process polygon types
    
    for(int typ=0; typ < 3; typ++) {
        
        int anzChilds = building.getChildCityObjectsCount();
        
        for(int p=0; p<anzChilds; p++) {
            const citygml::CityObject &buildingElement = building.getChildCityObject(p);
            std::string objtype = buildingElement.getTypeAsString();

            switch(typ) {
                case 0: if(objtype!="GroundSurface") continue;
                    break;
                case 1: if(objtype!="WallSurface") continue;
                    break;
                case 2: if(objtype!="RoofSurface") continue;
                    break;
            }
            //read coordinate list
            for(unsigned int i=0; i<buildingElement.getGeometriesCount(); i++) {
                
                int anzPolygons = buildingElement.getGeometry(i).getPolygonsCount();
                for(unsigned int k=0; k<anzPolygons; k++) {
                    
                    struct texturedPolygon_t polygon;
                    polygon.type=typ;
                    polygon.transformed=0;
                    
                    std::shared_ptr<const citygml::Polygon> citypolygon = buildingElement.getGeometry(i).getPolygon(k);
                    
                    polygon.id = building.getId();
                    std::shared_ptr<const citygml::LinearRing> ring= citypolygon->exteriorRing();
                    std::vector<TVec3d> ringvertices=ring->getVertices();
                    
                    //look for associated texture coordinates
                    std::vector<TVec2f> texcoords;
                    bool texture=false;
                    std::vector<std::string> themes = citypolygon->getAllTextureThemes(true);
                    if(themes.size()>0) {
                        texcoords = citypolygon->getTexCoordsForTheme(themes[0], true);
                        
                        //if there are interior polygons, the number of texture coordinates might exceed the number of outer vertices
                        while(texcoords.size()>ringvertices.size()) {
                            texcoords.pop_back();
                        }
                        if(texcoords.size()==ringvertices.size()) {
                            texture=true;
                            std::shared_ptr<const citygml::Texture> texture = citypolygon->getTextureFor(themes[0]);
                            polygon.texturename= texture->getUrl();
                            polygon.WrapMode=0;
                            citygml::Texture::WrapMode wrapmode = texture->getWrapMode();
                            if(wrapmode==citygml::Texture::WrapMode::WM_MIRROR) polygon.WrapMode = MIRROR;
                            else if(wrapmode==citygml::Texture::WrapMode::WM_CLAMP) polygon.WrapMode = CLAMP;
                            else if(wrapmode==citygml::Texture::WrapMode::WM_BORDER) polygon.WrapMode = BORDER;
                            //Berlin citymodel: NONE is wrongly used as WRAP
                            //else if(wrapmode==citygml::Texture::WrapMode::WM_NONE) polygon.WrapMode = NONE;
                        } else std::cout << "Warning: Number " << texcoords.size() << " of texture coordinates does not fit with number of vertices " << vertices.size() << "." << std::endl;
                    }
                    for(int j=0;j<ringvertices.size();j++) {
                        struct vertex_adj_t vertex;
                        vertex.p.x=ringvertices[j].x;
                        vertex.p.y=ringvertices[j].y;
                        vertex.p.z=ringvertices[j].z;
                        
                        cv::Point2d point;
                        if(texture) {
                            point.x=texcoords[j].x;
                            point.y=texcoords[j].y;
                        }
                        //does the vertex already exist?
                        unsigned int si=0;
                        for(si=0; si<vertices.size(); si++) {
                            if((fabs(vertices[si].p.x-vertex.p.x)<0.001)&&(fabs(vertices[si].p.y-vertex.p.y)<0.001)&&(fabs(vertices[si].p.z-vertex.p.z)<0.001)) break;
                        }
                        
                        if(si==vertices.size()) { //new vertex
                            vertices.push_back(vertex);
                        }
                        //Use vertex only if it does not equal its predecessor
                        if(polygon.polygon.size()>0) {
                            if(polygon.polygon[polygon.polygon.size()-1] != si) {
                                
                                //Is the predecessor an intermediate point?
                                
                                if(polygon.polygon_simplified.size()>1) {
                                    if(lieOnStraightLine(si, polygon.polygon_simplified[polygon.polygon_simplified.size()-1], polygon.polygon_simplified[polygon.polygon_simplified.size()-2],vertices)) {
                                        //update predecessor
                                        polygon.polygon_simplified[polygon.polygon_simplified.size()-1]=si;
                                        if(texture)
                                            polygon.texture_coordinates[polygon.polygon_simplified.size()-1]=point;
                                    } else {
                                        polygon.polygon_simplified.push_back(si);
                                        if(texture)
                                            polygon.texture_coordinates.push_back(point);
                                    }
                                } else {
                                    polygon.polygon_simplified.push_back(si);
                                    if(texture)
                                        polygon.texture_coordinates.push_back(point);
                                }
                                polygon.polygon.push_back(si);
                                struct allEdges_t edge;
                                edge.l=polygon.polygon[polygon.polygon.size()-2];
                                edge.r=si;
                                edge.polynr=(int)allPolygons.size();
                                allEdges.push_back(edge);
                            }
                        } else {
                            polygon.polygon.push_back(si);
                            polygon.polygon_simplified.push_back(si);
                            if(texture)
                                polygon.texture_coordinates.push_back(point);
                        }
                    }
                    //last vertex redundant?
                    if(polygon.polygon_simplified.size()>2) {
                        if(lieOnStraightLine(polygon.polygon_simplified[0], polygon.polygon_simplified[polygon.polygon_simplified.size()-1], polygon.polygon_simplified[polygon.polygon_simplified.size()-2],vertices))
                        {
                            //remove last vertex
                            polygon.polygon_simplified.pop_back();
                            if(texture)
                                polygon.texture_coordinates.pop_back();
                        }
                    }
                    //polygon[0] redundant?
                    if(polygon.polygon_simplified.size()>2) {
                        if(lieOnStraightLine(polygon.polygon_simplified[1], polygon.polygon_simplified[0], polygon.polygon_simplified[polygon.polygon_simplified.size()-1],vertices))
                        {
                            //remove first vertex
                            for(int i=1; i<polygon.polygon_simplified.size(); i++) {
                                polygon.polygon_simplified[i-1]=polygon.polygon_simplified[i];
                                if(texture)
                                    polygon.texture_coordinates[i-1]=polygon.texture_coordinates[i];
                            }
                            polygon.polygon_simplified.pop_back();
                            if(texture)
                                polygon.texture_coordinates.pop_back();
                        }
                    }
                    
                    //Additionally add first vertex as last vertex
                    if(polygon.polygon.size()>0) {
                        polygon.polygon.push_back(polygon.polygon[0]);
                        polygon.polygon_simplified.push_back(polygon.polygon_simplified[0]);
                        if(texture) {
                            polygon.texture_coordinates.push_back(polygon.texture_coordinates[0]);
                        }
                        struct allEdges_t edge;
                        edge.l=polygon.polygon[polygon.polygon.size()-2];
                        edge.r=polygon.polygon[0];
                        edge.polynr=(int)allPolygons.size();
                        allEdges.push_back(edge);
                    }
                   
                    
                    if(polygon.polygon.size()>3) {
                        //Inner polygons
                        std::vector<std::shared_ptr<citygml::LinearRing>> innerePolygone = citypolygon->interiorRings();
                        for(int ii=0; ii<innerePolygone.size(); ii++) {
                            std::vector<TVec3d> innervertices=innerePolygone[ii]->getVertices();
                            std::vector<unsigned int> interior;
                            for(int jj=0;jj<innervertices.size();jj++) {
                                struct vertex_adj_t vertex;
                                vertex.p.x=innervertices[jj].x;
                                vertex.p.y=innervertices[jj].y;
                                vertex.p.z=innervertices[jj].z;
                                
                                //check if vertex already exists:
                                unsigned int j=0;
                                for(j=0;j<vertices.size();j++) {
                                    if((fabs(vertices[j].p.x-vertex.p.x)<0.001)&&(fabs(vertices[j].p.y-vertex.p.y)<0.001)&&(fabs(vertices[j].p.z-vertex.p.z)<0.001)) break;
                                }
                                
                                if(j==vertices.size()) { //new vertex
                                    vertices.push_back(vertex);
                                }
                                
                                if(interior.size()>0) {
                                    if(interior[interior.size()-1]!=j) {
                                        interior.push_back(j);
                                    }
                                } else {
                                    interior.push_back(j);
                                }
                            }
                            //Additionally add first vertex as last vertex
                            if(interior.size()>2) {
                                interior.push_back(interior[0]);
                                polygon.interior_polygons.push_back(interior);
                            }
                        }
                    
                        //Compute normal
                        if(!computeNormal(polygon, vertices)) {
                            
                            //determine 2D plane of the polygon
                            computePlane(polygon, vertices);
                        
                            //transform 3D-coordinates to 2D-coordinates innerhalb der Polygonebene
                            polygonTo2DPlane(polygon, vertices, allPolygons);
                            //add a reference of the polygon to its vertices
                            int nr=(int)allPolygons.size();
                            for(int j=0;j<-1+(int)polygon.polygon.size(); j++) {
                                vertices[polygon.polygon[j]].inPolygon.push_back(nr);
                            }
                            for(int k=0; k<polygon.interior_polygons.size(); k++) {
                                for(int j=0;j<-1+(int)polygon.interior_polygons[k].size(); j++) {
                                    vertices[polygon.interior_polygons[k][j]].inPolygon.push_back(nr);
                                }
                            }
                            allPolygons.push_back(polygon);
                            
                        } else std::cout << "Warning: A face normal could not be computed, polygon degenerated." << std::endl;
                    } else std::cout << "Warning: Polygon with less than three vertices found." << std::endl;
                }
            }
        }
    }
    //Check if an edge is used more than once in the same direction
    bool first=true;
    
    std::sort(allEdges.begin(),allEdges.end(),compareEdges);
    
    for(int i=0; i<-1+(int)allEdges.size(); i++) {
        for(int j=i+1; j<allEdges.size() ; j++) {
            //Since the vector is sorted, one can stop when the first component changes
            if(allEdges[i].l != allEdges[j].l) break;
               
            if(allEdges[i].r == allEdges[j].r) {
                if(first) {
                    std::cout << "Warning: The model contains edges that are traversed more than once in the same direction: Intersecting polygons might occur." << std::endl;
                    first=false;
                }
                //Try to correct this by duplicating vertices
                if(allEdges[j].polynr<allPolygons.size()) {
                    //Replace vertices
                    for(int k=0; k<-1+(int)allPolygons[allEdges[j].polynr].polygon.size(); k++) {
                        if(allPolygons[allEdges[j].polynr].polygon[k]==allEdges[j].l) {
                            vertex_adj_t new_vertex;
                            new_vertex.p = vertices[allEdges[j].l].p;
                            new_vertex.inPolygon.push_back(allEdges[j].polynr);
                            vertices.push_back(new_vertex);
                            allPolygons[allEdges[j].polynr].polygon[k]=(int)vertices.size()-1;
                        } else if(allPolygons[allEdges[j].polynr].polygon[k]==allEdges[j].r) {
                            vertex_adj_t new_vertex;
                            new_vertex.p = vertices[allEdges[j].r].p;
                            new_vertex.inPolygon.push_back(allEdges[j].polynr);
                            vertices.push_back(new_vertex);
                            allPolygons[allEdges[j].polynr].polygon[k]=(int)vertices.size()-1;
                        }
                    }
                    allPolygons[allEdges[j].polynr].polygon[-1+(int)allPolygons[allEdges[j].polynr].polygon.size()]=allPolygons[allEdges[j].polynr].polygon[0];
                }
            }
        }
    }
}

void printAddress(const citygml::Address *printAdr) {
    
    if(printAdr==NULL) fprintf(latexfile,"\\\\\n");
    else {
        std::string streetName=printAdr->thoroughfareName();
        std::string number=printAdr->thoroughfareNumber();
            
        fprintf(latexfile,"\n\n\\noindent{}%s %s\\\\\n", streetName.c_str(), number.c_str());
    }
}

void processAllBuildings(std::shared_ptr<const citygml::CityModel> city)
{
    const citygml::ConstCityObjects& roots = city->getRootCityObjects();
    for ( unsigned int i = 0; i < roots.size(); i++ ) {
        const citygml::CityObject& object = *roots[i];
    
        std::string basistyp = object.getTypeAsString();
        if((basistyp!="Building")&&(basistyp!="BuildingPart")) continue;
    
        const citygml::Address *adr = object.address();
        int anzChilds = object.getChildCityObjectsCount();
        if(anzChilds==0) continue;
        
        const citygml::CityObject &firstObj = object.getChildCityObject(0);
        std::string firstObjtype = firstObj.getTypeAsString();
        if(firstObjtype=="BuildingPart") {
            
            //Building consists of building parts
            for(int p=0; p<anzChilds; p++) {
                const citygml::CityObject &buildingPartObj = object.getChildCityObject(p);
                std::string objtype = buildingPartObj.getTypeAsString();
                if(objtype!="BuildingPart") continue;
                
                //List of all vertices
                std::vector<struct vertex_adj_t> vertices;
                std::vector<texturedPolygon_t> allPolygons;
                
                //read different address
                const citygml::Address *localAdr = object.address();
                processBuilding(buildingPartObj, vertices, allPolygons);
                std::vector<struct fold_edge_t> foldEdges;
                
                const citygml::Address *printAdr = localAdr;
                if(printAdr==NULL) printAdr = adr;
                
                if(LATEX_OUTPUT) printAddress(printAdr);
                
                generateLayout(foldEdges, vertices, allPolygons);
                
            }
        }
        else {
            //no building parts
            if(LATEX_OUTPUT) printAddress(adr);
            
            std::vector<struct vertex_adj_t> vertices;
            std::vector<texturedPolygon_t> allPolygons;
            
            processBuilding(object, vertices, allPolygons);
            std::vector<struct fold_edge_t> foldEdges;
            
            generateLayout(foldEdges, vertices, allPolygons);
        }
    }
}

void usage()
{
    std::cout << "Usage: papermodel [-options...] <filename>" << std::endl;
    std::cout << " Options:" << std::endl;
    std::cout << "  -scale      <n>   Scale model by 1:n if 50 <= n <= 10000, default: n = 220" << std::endl;
    std::cout << "                    n = 0: only images are written but no LaTeX output" << std::endl;
    std::cout << "                    Scaling is done using LaTeX. Please execute command" << std::endl;
    std::cout << "                    \"pdflatex papermodel\" after running this tool to" << std::endl;
    std::cout << "                    obtain a PDF with correctly scaled building kits." << std::endl;
    std::cout << "  -resolution <n>   Texture pixels per meter, 1 <= n <= 100, default: n = 15" << std::endl;
    std::cout << "  -papersize  <n>   Generate DIN A<n> output, n = 0, 1, 2, 3, 4; default: A4" << std::endl;
    std::cout << "  -dest       <dir> Write LaTeX file and images to the folder <dir>" << std::endl;
}

int main(int argc, const char * argv[]) {
    
    scale=SCALE;
    resolution=RESOLUTION;
    output_LaTeX=LATEX_OUTPUT;
    papersize=PAPERSIZE;
   
    char inputfilename[1001], path[1001];

    if(argc<2) {
        usage();
        return 1;
    } else {
        int lastSlash=-1;
        const char *z=argv[argc-1];
        int i=0;
        while((i<1000)&&(*z!=0)) {
            if((*z=='/')||(*z=='\\')) lastSlash=i;
            inputfilename[i++]=*z++;
        }
        inputfilename[i]=0;
        i=0;
        char *zz=path;
        while(i<=lastSlash) {
            *zz++=inputfilename[i++];
        }
        *zz=0;
        path_textures=path;
        path_output=path;
        
        if(argc>2) {
            
            if(argc%2==1) {
                usage();
                return 1;
            }
            
            for ( int i = 1; i < argc-1; i+=2 )
            {
                std::string param = std::string( argv[i] );
                if ( param == "-scale" ) {
                    scale=stoi(std::string( argv[i+1] ));
                    if((scale!=0)&&((scale<50)||(scale>10000))) {
                        usage();
                        return 1;
                    } else {
                        if(scale==0) output_LaTeX = false;
                    }
                    
                } else if ( param == "-resolution" ) {
                    resolution=stoi(std::string( argv[i+1] ));
                    if((resolution<1)||(resolution>100)) {
                        usage();
                        return 1;
                    }
                } else if ( param == "-papersize" ) {
                    papersize=stoi(std::string( argv[i+1] ));
                    if((papersize<0)||(papersize>4)) {
                        usage();
                        return 1;
                    }
                } else if ( param == "-dest" ) {
                    path_output=std::string( argv[i+1] );
                    if(path_output.size()<1) {
                        usage();
                        return 1;
                    }
                    if((path_output.at(path_output.size()-1)!='\\') && (path_output.at(path_output.size()-1)!='/')) {
                        path_output += "/";
                    }
                } else {
                    usage();
                    return 1;
                }
            }
        }
    }
    
    //Compute thickness of lines depending on scale and resolution
    line_thickness = round(resolution/25.0);
    if(line_thickness<1) line_thickness=1;
    if(scale>0) {
        double factor=round(scale/600.0);
        if(factor>1) line_thickness*=factor;
    }
    //Compute maximum image size in pixels
    //If the image does not fit onto a seet of paper: divide it
    //paper size:
    //A4: 297 mm x 210 mm
    //A3: 420 mm x 297 mm
    //A2: 594 mm x 420 mm
    //A1: 840 mm x 594 mm
    //A0: 1188 mm x 840 mm
    double px, py;
    switch(papersize) {
       case 0: px=113.0; py=81.0; break;
       case 1: px=81.0; py=56.0; break;
       case 2: px=56.0; py=39.0; break;
       case 3: px=39.0; py=27.0; break;
       default: px=27.0, py=17.0;
    }
    if(scale!=0) {
        pixelwidth_x= (px * scale/100.0)*resolution;
        pixelwidth_y= (py * scale/100.0)*resolution;
    } else {
        pixelwidth_x=pixelwidth_y=10000000;
    }
    
    city = readModel(inputfilename);
    if(!city) {
        std::cout << "Could not read and parse " << inputfilename << std::endl;
        return 0;
    }

    if(output_LaTeX) {
        std::stringstream name;
        std::string buffer;
        name.str(std::string());
        name << path_output << "papermodel.tex";
        buffer = name.str();
        
        latexfile = fopen(buffer.c_str(),"wb");
        if(!latexfile){
           std::cout << "Could not write LaTeX file: " << buffer << std::endl;
           fclose(latexfile);
           return 0;
        }
        fprintf(latexfile, "\\documentclass[12pt,twoside, pdftex, paper=A%d,paper=landscape, pagesize]{scrartcl}\n",papersize);
        fprintf(latexfile, "\\oddsidemargin-20truemm\n\\evensidemargin-20truemm\n\\topmargin-20truemm\n");
        fprintf(latexfile, "\\usepackage{german}\\usepackage{graphicx}\\usepackage[utf8]{inputenc}\n\\pagestyle{empty}\n\\begin{document}\n");
        fprintf(latexfile, "\\noindent{Paper models scaled 1:%d generated from input file %s with a tool provided by iPattern institute, Niederrhein University of Applied Sciences}\\\\\n",(int)scale,inputfilename);
    }

    try {
        processAllBuildings(city);
    } catch(...) {
        std::cout << "An error occurred, please check parameters." << std::endl;
        return 0;
    }
        
    if(output_LaTeX) {
        fprintf(latexfile, "\\end{document}");
        fclose(latexfile);
        
    }
    return 0;
}
