% Week 8 (3/29): Model comparisons (Chang)

% 1. Reasons for model-fitting your data and selecting an appropriate model (concept) 

% 2. Comparing models with different number of parameters (concept)

% 3. Dealing with multiple comparisons (with practical component)

% 4. Fitting your data to different models and evaluating individual model fits

% 5. Comparing the model fits to find the best model (AIC, BIC, cross-validation)



%%%%%%%%%%%%%%%%%%%

% 1. Reasons for model-fitting your data and selecting an appropriate model (concept) 

% 2. Comparing models with different number of parameters (concept)

    % a) Testing HOW the system works - you have a model and using the model to help explain the data to gain deeper insights into the mehcanisms
    
    % b) Choosing a right model and a right set of parameters critically depend on what you know about the system 
         % (visual/motor tuning, reinforcement/learning, attention/normalization, etc)  
         
    % c) A famous scientist once said: "if you have 9 or more parameters, you can fit anything".  Basically, you want to minimize the number
         % of model parameters. 
         
         % Truth: more parameters always = better fitting.  Having too many parameters make your model fit meaningless.
         
         % Solution: Keep only the critical ones, and test the significant necessity of additional parameters if added 
         % (comparing Model with N parameter to Model with N-1 parameter)
         
    % d) Simpler model that can explain your data is the best


% 3. Dealing with multiple comparisons (with practical component)

    % a) Everytime you run a statistics test, you are more likely to find a positive result (false positive rate) -> multiple comparison problem
    
    % b) Many ways to get around with this:
        % i) Bonferroni correction (most conservative/most well known: familywise error correction): 
        
             % Basically, you adjust the alpha value by dividing it by the number of times you tested
             
             % e.g., if testing for p of 0.05 five times, the alpha needs to be adjusted to p of 0.01
             
             % same idea for one-tailed with P of 0.05 is two tailed with 0.025 (since you are comparing both tails)
             
             % Example:
              mu = [10 20 100 70];
              sigma = 40;
              z = repmat(mu,10,1) + randn(10,4)*sigma;

              b = [mean(z(:,1)) mean(z(:,2)) mean(z(:,3)) mean(z(:,4))];
              bar(b);
              hold on;
              line([1 1],[mean(z(:,1))-sigma mean(z(:,1))+sigma]);
              line([2 2],[mean(z(:,2))-sigma mean(z(:,2))+sigma]);
              line([3 3],[mean(z(:,3))-sigma mean(z(:,3))+sigma]);
              line([4 4],[mean(z(:,4))-sigma mean(z(:,4))+sigma]);
              
              % Let's find some significant differences
              % how many comparisons are you making? - DEPENDS on HYPOTHESIS.
              % Let's say that '1' is the baseline. So, we are testing 1 vs 2, 1 vs 3, 1 vs 4
              % That's 3 comparisons
              % So, to detect with p of 0.05, the p value needs to be
              % adjusted by '/3', p = 0.0167
              % 1 vs 2
              [a p]=ttest(z(:,1),z(:,2))
              p < 0.0125  
              % 1 vs 3
              [a p]=ttest(z(:,1),z(:,3))
              p < 0.0125  
              % 1 vs 4
              [a p]=ttest(z(:,1),z(:,4))
              p < 0.0125   % if not-corrected, it could be!
              
              % if you compare all the pairings, then you are making 6 comparisons, so for p of 0.05 -> p of 0.05/6 = 0.0083
              % pick your hypothesis carefully

        % 2) FDR (False discovery rate) correction (less strong than the familywise error correction)
        
            % specifies expected proportion of discoveries out of all discoveries 
        
            % P values from smallest to largest get different significance threshold.
            
            % Threshold = target_alpha * (1: length(P_values) ) * length(P_values)
            
            % e.g., 5 p values are obtained, then then thresholds for significance for each ORDERED p value are:
            
            % threshold for the smallest p = 0.05 * (1 / length(P_values)) [same as Bonferroni]
            
            % threshold for the 2nd p = 0.05 * (2 / length(P_values)) ...
            
            % threshold for the largest (5th) p = 0.05 * (5 / length(P_values))
        
            
            
% 4. Fitting your data to different models and evaluating individual model fits
%% Generate some decaying data over time - compare two types of decay models

% Let's generate some decaying data over time to fit two different well-known decay models
% (Reed & Martens, 2011)

days = [1, 5, 30, 60, 180, 270, 520, 1460];
value = [99, 70, 60, 50, 30, 20, 7, 1];

%   Model 1: Exponential decay model (2 parameters, A rate and B gain)

% in Matlab, @ is for handling functions
% @(params, x) means it's a function of x with params
% "params" is an array specifying values of parameters of ProbDistParametric object
exponential_func = @(params, x) params(1) .* exp(-params(2) * x);

%   Define starting values for the fitting parameters (need this for fitting anything)

exp_x0 = [0 0];

%   Obtain parameter estimates and normalized residuals based on the data
%   given in `days` and `value`
%   "lsqcurvefit" is a nonlinear least-squares solver for finding solutions that *converge*

%   x = lsqcurvefit(fun,x0,xdata,ydata) starts at x0 and finds coefficients x to best fit the 
%        nonlinear function fun(x,xdata) to the data ydata (in the least-squares sense).

[exp_params, exp_resnorm] = lsqcurvefit( exponential_func, exp_x0, days, value );

%   Model 2: Hyperbolic decay model (2 parameters, a rate and b gain)

hyperbolic_func = @(params, x) params(1) ./ (1 + params(2) .* x);

%   Again, define starting values for the parameters

hyp_x0 = [0 0];

%   Do the fitting

[hyp_params, hyp_resnorm] = lsqcurvefit( hyperbolic_func, hyp_x0, days, value );

%   Plot the real data and model fits

figure;
x_vector = min(days):max(days);

%   plot the real data with symbol *
plot( days, value, 'k*' ); hold on;  
%   plot the fit -> use the function we made (exponential_func) with the fitted parameters "exp_params" as a function of x_vector: 
%   This can be done by "exponential_func(exp_params, x_vector)" in the Y term for the plot
plot( x_vector, exponential_func(exp_params, x_vector), 'b' );
plot( x_vector, hyperbolic_func(hyp_params, x_vector), 'r' );
legend({ 'Data', 'Exponential Fit', 'Hyperbolic Fit'} );

%   Let's examine normalized residuals

figure;
bar( [exp_resnorm, hyp_resnorm] );  % low residuals are better
set( gca, 'xticklabels', {'Exponential', 'Hyperbolic'} );
ylabel( 'Normalized Residuals' );
xlim( [.5 2.5] );
% Hyperbolic does better as expected.

%% 
    
% 5. Comparing the model fits to find the best model (AIC, BIC, cross-validation)
    
    % a) AIC and BIC
        % AIC (Akaike information Criterion) and BIC (Bayesian Information Criterion) penalizes the parameter for the information lost.
        % AIC is an estimate for an unknown true likelihood function of the data and the fitted likelihood function
    
        % AIC = 2k - 2ln(L)
        % L is the maximized value of likelihood function of the Model: L = p(x | theta, Model), where theta is parameter value that maximizes L.
        % k is the number of free parameters (taking into account the number of parameters of the Model)
        % x is the data
        % Model with the minimum AIC value is the better model
        
        % BIC = k*ln(n) - 2ln(L)    % ln(e^x)=x
        % n is x's sample size
        % Model with the minimum BIC value is the better model
    
        % DIFFERENCE?  
        % BIC converges into one model with the growing number of observations (sample size). 
        % So, BIC is more conservative in picking a "smaller model" when the sample size is high 
        % - that is, BIC is penalizes the number of parameters more than AIC
       
    % b) Cross-validation
        % involves partitioning a sample of data into the training set and the validation set. 
        % Multiple rounds of cross-validation using different partitions to average the validation over the rounds.
        % Use the error between the training set and the validation set (the model with consistently smaller errors wins)

        
%% AIC example

% for AIC to work, you need a likelihood function

% R = poissrnd(lambda, m, n) generates random numbers from the Poisson distribution with mean parameter lambda. 
% m by n array
x = poissrnd(20,30,1);

% pd = fitdist(x,distname) creates a probability distribution object by fitting the distribution specified by distname 
% to the data in column vector x.
pd1 = fitdist(x,'poisson');
pd2 = fitdist(x,'normal');

% log likelihood for each model ("NlogL" field contains this info)
pd1LogL = -1*pd1.NLogL; 
pd2LogL = -1*pd2.NLogL;

% get AIC
% aic = aicbic(logL,numParam) returns Akaike information criteria (AIC) 
% corresponding to optimized loglikelihood function values (logL), 
% as returned by estimate, and the model parameters, numParam.
% look up aicbic function
numParam1=1; % poisson has 1 parameter, but normal has 2
numParam2=2;

aic = aicbic([pd1LogL, pd2LogL], [numParam1,numParam2]);
% poisson wins (lower AIC)

% BIC requires the number of observations as indicated above
% so let's try with different number of observations
x = poissrnd(20,30,1);
%y = poissrnd(20,130,1);  % much larger sample -> try different number of observations -> results change!

% pd = fitdist(x,distname) creates a probability distribution object by fitting the distribution specified by distname 
% to the data in column vector x.
pd1 = fitdist(x,'poisson');
pd2 = fitdist(x,'normal');

% log likelihood for each model
pd1LogL = -1*pd1.NLogL; 
pd2LogL = -1*pd2.NLogL;

[aic bic] = aicbic([pd1LogL, pd2LogL], [numParam1,numParam2], [30,30]);


%%  CROSS VALIDATION to compare model fits

%   Let's do cross-validation comparison (smaller root mean sqaure error, RMSE is better)
%   For: testing how the results of a statistical analysis will generalize to an independent data set.
%   Use a "training dataset" and test it on the "tested dataset"
%   Good for: not overfitting the data
%   Do this mutiple times (take care of noise/variability)

%   using the parameters of the hyperbolic equation, let's generate a
%   larger set of delay-discounted values

rng( 'default' ); % for reproducibility (random number generator)

%   generate a vector of days 

x = 1:1000;

%   obtain a vector of discounted values based on the hyperbolic function
%   and estimated hyperbolic parameters, above

% same as above, hyperbolic_func = @(params, x) params(1) ./ (1 + params(2) .* x);
% here we will generate a hyperbolic data first (then we will add noise to this to make "observed" data to test whether hyperbolic or exponential will do better)
% so, first generate hyperbolic values
y = hyperbolic_func( hyp_params, x );
 
%   add some NOISE to the values in y (this will generate "observed" data [for testing two models])
noise_gain = 3;
noise_index = rand( size(y) ) > .5;
% if there is 0, then replace it
noise_index = double(noise_index);
noise_index( find(noise_index==0) ) = -1;

% many ways to define the noise-added version, here is one way
% so, "y" is the data we are comparing the two models' performances here

noise = 1 ./ (1 + rand(size(noise_index))) * noise_gain;
y = y + noise_index .* noise;
% y = y + noise_index .* rand(size(noise_index)) * noise_gain;

n_partitions = 100;  % partitioning for cross-validation
partition_size = numel(y) / n_partitions;
start = 0;

%   preallocated index of whether, for the training sample, the hyperbolic
%   or exponential function had lower mse

hyperbolic_perfomed_better = false( size(n_partitions) );  % set up storing the outcome of the cross-validation

%   k-fold cross-validation (one type of cross-validation)

for i = 1:n_partitions  % we set it at 10 parts (so, testing 10 parts of the data)
  
  keep_index = true( size(y) );
  terminus = start + partition_size;  % for this round, first "part" of the dataset
  start = start + 1;
  keep_index( start:terminus ) = false;   % setting up indexing (1 or 0) for one set
  kept_x = x( keep_index );
  kept_y = y( keep_index );
  left_x = x( ~keep_index );
  left_y = y( ~keep_index );
  
  % STEP 1
  % Use the "kept dataset, kept_x and kept_y" to derive the model exp and hyp model parameters
  exp_params = lsqcurvefit( exponential_func, exp_x0, kept_x, kept_y ); 
  hyp_params = lsqcurvefit( hyperbolic_func, hyp_x0, kept_x, kept_y);
  
  % STEP 2
  % Now, how use the fit parameters based on "kept dataset" above work with "left x" to "*predict left y" 
  
  % -> The fitted values "exp_params" from the "kept datasets" (exp_parms and hyp_params) are now
  %    expressed as a function of "left_x" to "*predict left_y"
  predicted_y_exp = exponential_func( exp_params, left_x );
  predicted_y_hyp = hyperbolic_func( hyp_params, left_x );

  % STEP 3
  % You have real "left y" and "predicted left y" from the model parameters based on the kept dataset ("kept_x" and "kept_y")
  
  % Get mean of the squared errors for each model: 1) predicted_y from the model constructed 
  % From the "kept dataset" VS. 2) true y of the "left dataset, left_y"
  mse_exp = mean( (predicted_y_exp - left_y).^2 );
  mse_hyp = mean( (predicted_y_hyp - left_y).^2 );
  
  % now compare the model and save the outcome
  hyperbolic_perfomed_better(i) = mse_hyp < mse_exp;
  
  % start from where we left off and go to the next "part" of the data
  start = terminus;
  
end

p = 1 - sum(hyperbolic_perfomed_better) / n_partitions;

disp( p );


%% End of Class            
        
        
        