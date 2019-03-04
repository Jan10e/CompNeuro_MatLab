function [y, prob] = simBinary(stim, beta)

% generate binary choice data based on stimulus strength; stim - matrix, each

% response probability 
prob = 1./(1+exp(-stim*beta));

% actual data 
noise = rand(size(prob)); 
y = prob>noise;

end

