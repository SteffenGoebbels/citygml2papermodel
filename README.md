# citygml2papermodel
**Unfolding of textured CityGML buildings to paper models**

Created by Steffen Goebbels on 11.02.2022, iPattern Institute, Faculty of Electrical Engineering and Computer Science, Niederrhein University of Applied Sciences, Krefeld, Germany, steffen.goebbels@hsnr.de

The algorithm is described in the paper

**Steffen Goebbels, Regina Pohle-Froehlich: Automatic Unfolding of CityGML Buildings to Paper Models. Geographies 1 (3), 2021, pp. 333–345, https://doi.org/10.3390/geographies1030018**

You can use and modify this program without any restrictions if you cite this paper. The code comes without any warranty or liability, use at your own risk.

Prerequisites: **OpenCV** and **libCityGML** (tested with version 2.3.0) libraries, **LaTeX** for PDF creation

The command line tool unfolds textured CityGML buildings to images that are written to the folder of the input data or to a folder specified by -dest. Also, a LaTeX source file "citymodel.tex" is generated that includes the images. Using the command "pdflatex citymodel", a PDF can be generated in which the unfolded models are correctly scaled with respect to the -scale parameter. The tool processes wall, roof, and ground plane polygons. The assumption is that their vertices are organized counter-clockwise, i.e., in the mathematically positive sense, when seen from outside the building. If this assumption is violated, polygons might overlap in the output. The tool only processes ground plane, wall and roof polygons. Textures are mapped using either an affine or a perspective transform. Whereas texture coordinates outside the interval [0, 1] are supported by periodically repeating (WRAP) as well as repeating and mirroring (MIRROR), a general texture transform like implemented in OpenGL is not applied. Currently, four paper sizes are supported: DIN A0 - A4. If a layout image does not fit onto a single page of paper, the image is divided into parts. Colored bars at the boundaries indicate how the parts fit together.
