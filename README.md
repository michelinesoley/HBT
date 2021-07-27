# HBT

TT-SOFT code for determination of UV pump/X-ray probe UV spectra
 and propagation of HBT with fully quantum treatment of all 69 degrees of freedom.
 
 # Dependencies
 
 This package requires the following:
 
 - python
 - ttpy
 - matplotlib
 - numpy
 - scipy
 - gnuplot (optional)
 
 # Instructions
 
 To run the dynamics code, enter the DynamicsCode folder and execute the following commands:
 
 ```
 python 2d_interpolator.py
 python ttmn.py
 python hbtinformation.py
 python snapshots.py
 ```
 
Further results can be visualized with the following commands:
 
 ```
 gnuplot norm.gpt
 gnuplot populations.gpt
 gnuplot energy.gpt
 gnuplot expectationvalueCCC.gpt
 gnuplot expectationvalueOH.gpt
 ```
 
 Additional documentation is included in documentation.pdf (under construction).
 

