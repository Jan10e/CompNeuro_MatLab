% MATLAB code for Week 6 class in INP599
% April 5, 2017
% Alex Kwan
%
% Week 9: Time series and spectral analysis

clear all;
close all;

%% Examples of periodicity in neuroscience
%powerpoint slides 2 - 3

%% Sine and cosine
figure;

ampl = 1;   %amplitude
period = 2; %period

t=[0:0.01:5];
f1=ampl*sin(2*pi*t/period);
plot(t,f1);

f2=ampl*cos(2*pi*t/period);
hold on;
plot(t,f2);

%% Sum of sines and cosines
figure;

f1=sin(2*pi*t);
plot(t,f1);
hold on;

%sum of two sines
f2=1/3*sin(2*pi*t *3);
plot(t,f1+f2);

%sum of three sines
f3=1/5*sin(2*pi*t *5);
plot(t,f1+f2+f3);

%sum of many sines to approximate squares
f4=sin(2*pi*t);
for n=3:2:49
    f4=f4+1/n*sin(2*pi*t *n);
end
plot(t,f4);

%Gibbs phenomenon

%Any curve can be approximated by a sum of sines and cosines, aka Fourier
%series. The tricky part is to find the coefficients

%% From Fourier series to Fourier transforms

% powerpoint slides 5 - 13

%% Data: extracellular recordings from neonatal mouse spinal cord during fictive locomotion
% for details regarding the experiment, see Kwan AC et al., J Neurophysiol, 2010
% left: [time x episode], recording of left ventral root for three episodes
% right: [time x episode], the corresponding recording of right ventral root
% t: [time x 1], time stamps

% powerpoint slide 14
clear all;
load spinalcord.mat;

figure;

for n=1:3
    subplot(2,3,n);
    plot(t,left(:,n)); %left ventral root
    xlabel('Time (s)');
    ylabel('Ipsilateral ventral root');
    axis tight;
    
    subplot(2,3,3+n);
    plot(t,right(:,n)); %right ventral root
    ylabel('Contralateral ventral root');
    axis tight;
end

%zoom in and out to look at the structure of the data

%% Calculating power spectrum using built-in functions
figure;

y=abs(left(:,1)); %rectify to focus on the envelope, rather than the high-frequency components

T=nanmean(diff(t)); %sampling frequency for this data set is 5 kHz
Fs=1/T;
L=numel(y);

f=Fs/L*(0:(L/2));
a=fft(y);
P=a.*conj(a)/L;

plot(f,P(1:L/2+1));
xlabel('Frequency (Hz)');
ylabel('Spectral power density');
xlim([0 5]);

%crosshair to check frequency of peak power spectral density

%using built-in functions from Matlab requires careful accounting of the
%normalization factors

%% Calculating power spectrum using Chronux
% will use the Chronux package for multi-taper spectral analysis
% download from chronux.org
% unzip, set path, add with subfolders
figure;

NW=1;
params.tapers=[NW 2*NW-1];
params.pad=0;  %padding to nearest power of 2
params.Fs=1/T;
params.fpass=[0 5];
params.err=[1 0.05];   %1 for theoretical error bars, p = 0.05
[S,f,Serr]=mtspectrumc(y,params);

plot(f,S);
xlabel('Frequency (Hz)');
ylabel('Spectral power');

%%
%logarithmic scale for y-axis
figure;
subplot(2,2,1);
semilogy(f,S,'r','LineWidth',3);
hold on; semilogy(f,Serr(1,:),'k'); semilogy(f,Serr(2,:),'k');
title('Single episode');
xlabel('Frequency (Hz)');
ylabel('Spectral power');

%% Trial-averaging
y=abs(left);

params.trialave=1;

[S,f,Serr]=mtspectrumc(y,params);

subplot(2,2,2);
semilogy(f,S,'r','LineWidth',3);
hold on; semilogy(f,Serr(1,:),'k'); semilogy(f,Serr(2,:),'k');
title('Trial averaging');
xlabel('Frequency (Hz)');
ylabel('Spectral power');

%% Increase the number of tapers
y=abs(left(:,1));

NW=2;
params.tapers=[NW 2*NW-1];
params.trialave=0;
[S,f,Serr]=mtspectrumc(y,params);

subplot(2,2,3);
semilogy(f,S,'r','LineWidth',3);
hold on; semilogy(f,Serr(1,:),'k'); semilogy(f,Serr(2,:),'k');
title('#Tapers = 3');
xlabel('Frequency (Hz)');
ylabel('Spectral power');

%% Spectrogram - time-frequency analysis
figure;

y1=abs([left(:,1); left(:,2); left(:,3)]);   %concatenate the episodes
y1=y1-mean(y1);    %remove DC bias

movingwin=[5 0.5];    %moving window [length step_size], in units of Fs
NW=1;
params.tapers=[NW 2*NW-1];
[S,time,f]=mtspecgramc(y1,movingwin,params);

imagesc(time,f,S');
set(gca,'Ydir','normal');   %so zero for y-axis starts from below
xlabel('Time (s)');
ylabel('Frequency (Hz)');

%% Coherence
% powerpoint slide 15
figure;

NW=2;
params.tapers=[NW 2*NW-1];

params.err=[2 0.01];   %2 for jackknife error bars, p = 0.05
params.trialave=1;

[C,phi,S12,S1,S2,f,confC,phistd,Cerr]=coherencyc(abs(left),abs(right),params);

figure;
subplot(2,1,1);
plot(f,C);  %magnitude of coherence
hold on;
plot(f,Cerr(1,:),'k');
plot(f,Cerr(2,:),'k');
plot([f(1) f(end)],confC*[1 1]);    %CI corresponding to p=0.05
xlabel('Frequency (Hz)');
ylabel('Coherence magnitude');

subplot(2,1,2);
plot(f,phi); %phase of coherence, unwrap the phase angles
xlabel('Frequency (Hz)');
ylabel('Coherence phase');

%phase at peak coherence magnitude is ~3; 3/2pi = 0.47 portion of cycle

%% Autocorrelation and cross-correlation
figure;

y=abs(left(:,1));
[r,lag]=xcorr(y);
subplot(2,1,1);
plot(lag*T,r);
xlim([-5 5]);
xlabel('Lag (s)');
ylabel('Autocorrelation');

yy=abs(right(:,1));
[r2,lag2]=xcorr(y,yy);
subplot(2,1,2);
plot(lag2*T,r2);
xlim([-5 5]);
xlabel('Lag (s)');
ylabel('Cross-correlation');

%peak autocorr at time of 1.88 s
%peak cross-corr at time of 0.88 s

%0.88/1.88 ~0.47, same as estimate from coherence

%% Simulated data: Gaussian random walk
clear all;
figure;

%Gaussian random walk with mean = 0, std = 1
T=0.001;     %sampling interval
t=T*(1:2^16);    %time
w(1)=0;
for n=2:numel(t)
    w(n)=w(n-1)+randn;
end
subplot(2,2,1);
plot(t,w,'k');
xlabel('Time (s)');

Fs=1/T;
L=numel(w);
f=Fs/L*(0:(L/2));
a=fft(w);
P=a.*conj(a)/L;
S=P(1:L/2+1);

subplot(2,2,2);
loglog(f,S,'k');
xlabel('Frequency (Hz)');
ylabel('Spectral power density');
xlim([0.1 100]);

stats=regstats(log(S(500:5000)),log(f(500:5000)),'linear');
stats.beta

%exponent is 2; 'Brown noise'
% powerpoint slide 17 - 18

%% Low-pass filter

% powerpoint slide 19 - 21

%find index corresponding to 2 Hz
[~,idx]=min(abs(f-2));

%remove all the spectral content above 2 Hz
amod=zeros(size(a));
amod(1:idx)=a(1:idx);
amod(end-idx:end)=a(end-idx:end);

subplot(2,2,3);
loglog(abs(a),'k');
xlim([10 idx*10]);
hold on;
loglog(abs(amod),'r');

%inverse Fourier transform
wmod=ifft(amod);

subplot(2,2,4);
plot(t,w,'k');
hold on;
plot(t,real(wmod),'r');
xlabel('Time (s)');

%effectively we smoothed the random walk signal


