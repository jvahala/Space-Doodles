%% DoodleVerse_Lab_q1q2sol.m 
% DoodleVerse Project Solutions, Lab questions 1-2

clear; 
clc;
close all

% read image and convert to grayscale (black and white)
imgC = imread('diamond.png');
imgG = rgb2gray(imgC);
simg = size(imgG);

% pad with white space all around to do convolution
imgGpad = padarray(imgG,[1 1], 255);

%define simple edge detector filter, sobel operator is also a good choice
filter = [-1 0 1];

% do convolution with edge detector filter
img = conv2(single(imgGpad),single(filter));

%first and last few columns are bogus due to edge border, first and last rows are padding
xset = 1:simg(1);
yset = 3:simg(2)-3;

%look for edges, then make matrix of the index locations
[I,J] = find(abs(img(xset,yset)) > 150);

% add 1 to account for missed pixel around the border in the png
I = I + 1;
J = J + 1; 
f_ind = [I J]; %f_ind represents your x,y coordinates of perfect border locations
num_features = length(f_ind)

% this is too many features, need to reduce somehow, try taking only
% the extremes
left_most = [min(f_ind(:,1)), f_ind(find(f_ind(:,1) == min(f_ind(:,1)),1),2)];
right_most = [max(f_ind(:,1)), f_ind(find(f_ind(:,1) == max(f_ind(:,1)),1),2)];
bottom_most = [f_ind(find(f_ind(:,2) == min(f_ind(:,2)),1)), min(f_ind(:,2))];
top_most = [f_ind(find(f_ind(:,2) == max(f_ind(:,2)),1)), max(f_ind(:,2))];

features = [left_most;top_most;right_most;bottom_most]
features_plot = [features;left_most]; %adds left most point to connect line in plot

% plot to see how good it represents the shape
figure(1)
plot(features_plot(:,1),features_plot(:,2),'x-b');
axis([0 400 100 500]);