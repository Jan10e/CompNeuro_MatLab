%% load the digits

data = load('digits.mat');

%% plot some individual numbers

figure;
for ii=[1:16]
    subplot(4,4,ii);
    imagesc(reshape(data.dig5(ii,:),[28 28])');
    set(gca,'dataa',[1 1 1]);
    colormap('gray');
end

%% check it out

figure;
for ii=[0:9]
    subplot(3,4,ii+1); 
    eval(['imagesc(reshape(mean(data.dig' num2str(ii) ',1),[28 28])'');']);
    set(gca,'dataa',[1 1 1]);
    colormap('gray');
end

%% make giant matrix X with all numbers

X=[];
num=[];
for ii=[0:9]
    eval(['X = [X;data.dig' num2str(ii) '];']);
    num=[num;ones(400,1)*ii];
end

%% plot mean of the new ones

figure; imagesc(reshape(mean(X,1),[28 28])'); colormap('gray');

%% mean subtract X, compute the covariance matrix

X = double(X);
Xsave = X; % save it for later...
Xm = X - ones(size(X,1),1)*mean(X,1);
C = Xm'*Xm;
figure; imagesc(C); colorbar;

%% find the eigenvectors and eigenvalues of this covariance matrix

[v,d]=eig(C);

d =diag(d); d = d/sum(d); 

% it's all in reverse order
d=d(end:-1:1);
v = v(:,end:-1:1);

figure; plot(d,'.-'); xlabel('eigen vector index'); ylabel('fraction of variance accounted for');
set(gca,'yscale','log');

figure; plot(cumsum(d)); xlabel('eigen vectors included'); ylabel('fraction of variance accounted for');

%% plot the first few eigenvectors, just to check them out

figure
for ii=1:9
    subplot(3,3,ii);
    imagesc(reshape(v(:,ii),[28 28])'); colormap('gray'); colorbar;
    title(['eivenvector ' num2str(ii)]);
end

%% project all the points into the first two eigenvectors!

proj = Xm*v(:,1:2);

figure; hold on;
cm = colormap('lines');
for ii=0:9
    ind = [ii*400+1:(ii+1)*400];
%     plot(proj(ind,1),proj(ind,2),'.','color',cm(ii+1,:));
    plot(proj(ind,1),proj(ind,2),'.','color',rand(1,3),'markersize', 20);
end
title('all the digits');

%% try just plotting digits 0 and 1

figure; hold on;
cm = colormap('lines');
for ii=0:1
    ind = [ii*400+1:(ii+1)*400];
%     plot(proj(ind,1),proj(ind,2),'.','color',cm(ii+1,:));
    plot(proj(ind,1),proj(ind,2),'.','color',rand(1,3),'markersize', 20);
end
title('digits 0 and 1 are easily separable');

%% try just plotting digits 6 and 8

figure; hold on;
cm = colormap('lines');
for ii=[6,8]
    ind = [ii*400+1:(ii+1)*400];
%     plot(proj(ind,1),proj(ind,2),'.','color',cm(ii+1,:));
    plot(proj(ind,1),proj(ind,2),'.','color',rand(1,3),'markersize', 20);
end
title('digits 6 and 8 are NOT easily separable');

%% do PCA with just these digits

f=find((num==6) + (num==8));
X68=X(f,:);

X68m = X68 - ones(size(X68,1),1)*mean(X68,1);
C68 = X68m'*X68m;

[v68,d68]=eig(C68);
d68 =diag(d68); d68 = d68/sum(d68); 
d68=d68(end:-1:1);
v68 = v68(:,end:-1:1);

proj68 = X68m*v68(:,1:2);

figure; hold on;
cm = colormap('lines');
for ii=[0 1] % these are really 6 and 8
    ind = [ii*400+1:(ii+1)*400];
%     plot(proj(ind,1),proj(ind,2),'.','color',cm(ii+1,:));
    plot(proj68(ind,1),proj68(ind,2),'.','color',rand(1,3),'markersize', 20);
end
title('digits 6 and 8 are now pretty easily separable');

%% now, just for fun, let's run LDA on 6 and 8, and we can see how much better it looks

% turns out C68 has zero eigenvalues, so it's not easily invertible. to deal
% with this, we'll add elements along its diagonal...

C68r = C68 + eye(size(C68))*mean(d68);

w = (mean(X68(1:400,:),1)-mean(X68(401:800,:),1))/C68r;
p = X68*w';

figure;
subplot(2,1,1);
hist([proj68(1:400,1),proj68(401:800,1)],100);
xlabel('PCA 1');
ylabel('count');
title('digits 6 and 8 are not easily separable by PCA');

subplot(2,1,2);
hist([p(1:400),p(401:800)],100);
xlabel('LDA');
ylabel('count');
title('digits are more easily separable using LDA');

%% btw, of course we can also view each pixel as a data point, and each handwritten
%% digit as a feature; in other words, caring about X transpose; let's repeat
%% a bunch of what we did but with X' instead

X = Xsave';

Xm = X - ones(size(X,1),1)*mean(X,1);
C = Xm'*Xm;
figure; imagesc(C); colorbar;
% note: dimensions are now the digit index, not pixel index;

%% this may take a little while, since it's a big covariance matrix
[v,d]=eig(C);

d =diag(d); d = d/sum(d); 

% it's all in reverse order
d=d(end:-1:1);
v = v(:,end:-1:1);

%% now, plot some of this stuff up

figure; plot(d,'.-'); xlabel('eigen vector index'); ylabel('fraction of variance accounted for');
set(gca,'yscale','log');

figure; plot(cumsum(d)); xlabel('eigen vectors included'); ylabel('fraction of variance accounted for');

figure;
for ii=1:9
    subplot(3,3,ii);
    plot(v(:,ii));
    title(['eivenvector ' num2str(ii)]);
end



