%% loads and then plays with Susumu's data

a=xlsread('susumu_traces.xlsx');

%{
From Susumu on this code:

Each well is transfected with one gene. so difference is a gene transfected. Basically, 384 distinct genes are transfected on this plate.

Agonist-evoked ion channel activity was measured using calcium dye.
When you plot with time/response, you will see responses from 10-20 sec that was the time agonists were added to the all well simultaneously.

%}

%% plot it up...

t = a(:,1);
data = a(:,2:end);
dm = mean(data,2);

figure; hold on;
plot(t,data,'color',[0.5 0.5 0.5]);
plot(t,dm,'k-','linewidth',2);
xlabel('time (s)');
ylabel('fluorescence (a.u.)');

 


