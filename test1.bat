set path1=.\data\Image1.png
set path2=.\data\wsi.png
set dname_out=.\output\

mkdir %dname_out%

.\bin\Release\test_cv_matching1.exe %path1% %path2% %dname_out%
