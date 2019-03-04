% script for game of life
% by Daeyeol Lee
% Jan 26, 2017

maxuniv=input('Size of the universe = ');
plife=input('Probability of life = ');
maxtime=input('Duration of universe = ');
% initialize the universe
x=rand(maxuniv)<plife;

% run the uinverse
for k=1:maxtime,
    imagesc(x);
    x=update_life_universe(x);
    pause(0.1);
end;
