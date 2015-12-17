#README.txt

ECE 532 Lab - The DoodleVerse
Authors: Devin Conathan, Zach Pace, Josh Vahala

All necessary documents are included in the .zip provided

LAB REPORT DOCUMENTATION
---------------------------
1. lab_writeup.pdf - contains final report writeup including all solutions in boxes beneath each question
2. DoodleVerse_Lab_q1q2sol.m - MATLAB answer to Lab questions 1 and 2
3. lab_solution.py - python script for Lab questions 3 through 5
4. procrustes_solution.py - python script for answering procrustes warmup problem
5. procrustes.mat - data file needed to answer lab questions
6. clustercenters.mat - data file needed to answer lab questions
7. starlabels.mat - data file needed to answer lab questions
8. stars.mat - datafile needed to answer lab questions
9. diamond.png - diamond shape to be use for the lab
10. grading.txt - file containing grades for the three students assigned to our lab

DOODLEVERSE DOCUMENTATION - files needed to play with the DoodleVerse but not for the lab itself
---------------------------
11. doodleverse.py - main program to run the doodleverse
11a. Open doodleverse.py and change the file in main() to the file you would like to convert into a constellation. The current program only works for fully closed shapes (like those provided here). 
11b. If not working as you would like, play with the clustThresh value by making it nonnegative and between 0 and ~100. Larger values will remove more points. Making it negative activates a built in threshold based upon the shape itself. 
12. feature_extract.py - program needed to convert images into feature point sets
13. clustersearch.py - program needed to convert feature points into star points
14. hyg_catalog.fits - star data set

IMAGES PROVIDED - extra images to play with in doodleverse.py
---------------------------
15. mj.png - Michael Jackson outline
16. shape1.png - Duck? shape
17. shape2.png - Mountain Range? shape
18. shape3.png - Circle? shape
19. shape5.png - Swordsman? shape