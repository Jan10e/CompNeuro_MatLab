%% load in the data

load('probeEpochData.mat');
load('croppedMovie.mat');

%% plot some basics

figure;
subplot(2,1,1);
plot(probeEpochData);
ylabel('stimulus type');
xlabel('frame number');

subplot(2,1,2);
imagesc(squeeze(ims(:,:,1)));
colormap('gray');
axis image;
title('first image');
xlabel('pixel');
ylabel('pixel');

%% now, let's play ourselves a movie

figure;
for ii=1:size(ims,3)
    imagesc(ims(:,:,ii));
    set(gca,'clim',[0 max(ims(:))]);
    colormap('gray');
    title(['frame number = ' num2str(ii)]);
    axis image;
    drawnow;
end

%% let's just look at the a corner of the pixels

ims(1:10,1:5,1)

%% plot out a histogram of pixel values to see bit depth, and look for saturation

figure;
hist(ims(:),200);
set(gca,'yscale','log');
xlabel('pixel value');
ylabel('count through movie');



%% find the mean image over the entire stack, and find the std image over the entire stack
% this will make things a bit clearer, since individual movie frames are
% pretty noisy

imMean = mean(ims,3);
imSTD = std(ims,[],3);

figure; 
subplot(2,1,1);
imagesc(imMean);
title('mean');
colormap('gray');
subplot(2,1,2);
imagesc(imSTD);
colormap('gray');
title('std');

%% just for fun, plot the variance against the mean

% there's extra variability from the response here...
figure;
subplot(2,1,1);
plot(imMean(:),imSTD(:).^2,'r.');
xlabel('mean pixel value');
ylabel('var pixel value');
subplot(2,1,2);
plot(imMean(:),imSTD(:).^2./imMean(:),'r.');
xlabel('mean pixel value');
ylabel('var/mean');

%% now, let's start looking at what we can do here...

imMean(1:10,1:5) % check out the raw, mean pixel values

%%
% and now check out a histogram of those values
figure;
hist(ims(:),200);
xlabel('pixel value');
ylabel('count');
set(gca, 'yscale','log');

%% let's try out some linear filtering...

A = ones(10); % can also make more complicated filters -- round or gaussian. try one out here?
imf = conv2(imMean,A,'same');

figure;
subplot(2,1,1);
imagesc(imMean);
set(gca,'clim',[0 max(imMean(:))]);
colormap('gray');
axis image;
subplot(2,1,2);
imagesc(imf);
set(gca,'clim',[0 max(imf(:))]);
colormap('gray');
axis image;

%% highlight x-direction gradients!

gx = [1 0 -1; 2 0 -2; 1 0 -1];

imf = conv2(imMean,gx,'same');
figure;
subplot(2,1,1);
imagesc(imMean);
set(gca,'clim',[0 max(imMean(:))]);
colormap('gray');
axis image;
subplot(2,1,2);
imagesc(imf);
set(gca,'clim',[min(imf(:)) max(imf(:))]);
colormap('gray');
axis image;

%% highlight all edges/gradients

gx = [1 0 -1; 2 0 -2; 1 0 -1];
gy = [1 2 1; 0 0 0; -1 -1 -1];

imf = sqrt(conv2(imMean,gx,'same').^2 + conv2(imMean,gy,'same').^2);
figure;
subplot(3,1,1);
imagesc(imMean);
set(gca,'clim',[0 max(imMean(:))]);
colormap('gray');
axis image;
subplot(3,1,2);
imagesc(imf);
set(gca,'clim',[0 max(imf(:))]);
colormap('gray');
axis image;
subplot(3,1,3);
imagesc(log(imf));
set(gca,'clim',[0 max(log(imf(:)))]);
colormap('gray');
axis image;

%% instead of averaging, what about nonlinear filtering?

% first, let's just mess with gamma.
imMean1 = imMean/max(imMean(:));
G = [0.25 0.5 2 4];
figure;
subplot(5,1,1);
imagesc(imMean1);
set(gca,'clim',[0 1]);
colormap('gray');
axis image;

for ii=1:length(G)
    subplot(5,1,ii+1);
    imagesc(imMean1.^G(ii));
    ylabel(['\gamma = ' num2str(G(ii))]);
    set(gca,'clim',[0 1]);
    colormap('gray');
    axis image;
end
% note: screens all do this automatically

%% side note: color maps and limits make a huge difference

figure;
h1 = subplot(3,1,1);
imagesc(imMean);
set(gca,'clim',[0 max(imMean(:))]);
colormap(h1, 'gray');
axis image
colorbar;

h2 = subplot(3,1,2);
imagesc(imMean);
set(gca,'clim',[0 max(imMean(:))/5]);
colormap(h2,'gray');
axis image
colorbar;

subplot(3,1,3);
imagesc(imMean);
axis image
colorbar;


%% very useful way to filter out salt-and-pepper noise: median filter

imsel = ims(:,:,160);
figure;
subplot(2,1,1);
imagesc(imsel);
set(gca,'clim',[0 max(imsel(:))]);
axis image;
% pause;
subplot(2,1,2);
imagesc(medfilt2(imsel,[3 3]));
set(gca,'clim',[0 max(imsel(:))]);
axis image;


%% let's try thresholding this thing to see what we find

ims10 = ims > 1000;

% check out the movie
figure;
for ii=1:size(ims10,3)
    imagesc(ims10(:,:,ii));
    set(gca,'clim',[0 1]);
    colormap('gray');
    title(['frame number = ' num2str(ii)]);
    drawnow;
end

%% now, how to select regions of interest for further analysis?

im10 = (imMean > 500);
% note: you can find a good, rational threshold by k-means clustering or LDA!

figure;
subplot(2,1,1);
imagesc(imMean);
set(gca,'clim',[0 max(imMean(:))]);
colormap('gray');
axis image;
subplot(2,1,2);
imagesc(im10);
set(gca,'clim',[0 1]);
colormap('gray');
axis image;

%% now, some useful operations for binary images

im10d = im10;
figure;
for ii=1:5
    subplot(5,1,ii);
    im10d = imerode(im10d,ones(3));
    imagesc(im10d);
    set(gca,'clim',[0 1]);
    colormap('gray');
    axis image;
end

im10d = im10;
figure;
for ii=1:5
    subplot(5,1,ii);
    im10d = imdilate(im10d,ones(3));
    imagesc(im10d);
    set(gca,'clim',[0 1]);
    colormap('gray');
    axis image;
end

%% watershedding to find regions...

L = watershed(-conv2(imMean,ones(9)));
figure;
subplot(2,1,1);
imagesc(imMean);
axis image;
subplot(2,1,2);
imagesc(L);
axis image;

%% go back to binary and find all the connected objects

cc = bwconncomp(im10,ones(3));
cc
R = zeros(size(im10));
for ii=1:cc.NumObjects
    R(cc.PixelIdxList{ii})=ii;
end
figure;
imagesc(R);
axis image;

%% get region props

pp = regionprops(im10); % many options here -- see website
pp

%% now, how about we find out what the fluorescence of each cell is over time?

for ii=1:cc.NumObjects
    for tt=1:size(ims,3)
        temp = ims(:,:,tt);
        ff(ii,tt) = mean(temp(cc.PixelIdxList{ii}));
        % QUESTION: if you are measuring photons, is equal weighting the
        % way to go here, or should you weight some pixels more?
    end
end
figure;
plot(ff');
xlabel('time (frames)');
ylabel('fluorescence (au)');

%% plot up a nice figure of the results

figure;
subplot(2,1,1);
plot(probeEpochData);
ylabel('stimulus type');
subplot(2,1,2); hold on;
plot(mean(ff,1),'k');
plot(squeeze(mean(mean(ims,2),1)),'r');
legend('ROIs','full screen');
xlabel('time (frames)');
ylabel('fluorescence (au)');

%% just to check on the background... since we're using a light stimulus

figure;
imagesc(imMean);
M = roipoly;
idx = find(M);
for tt=1:size(ims,3)
    temp = ims(:,:,tt);
    bg(tt) = mean(temp(idx));
end

figure;
subplot(2,1,1);
plot(probeEpochData);
ylabel('stimulus type');
subplot(2,1,2);
plot(bg);
xlabel('time (frames)');
ylabel('background intensity');
% clearly some bleedthrough


