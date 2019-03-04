function test_effect(mCtrl, fCtrl, mDrug, fDrug)

data = [mCtrl fCtrl mDrug fDrug];
 
treatment = [zeros(size(mCtrl)) zeros(size(fCtrl)) ones(size(mDrug)) ones(size(fDrug))];

sex = [zeros(size(mCtrl)) ones(size(fCtrl)) zeros(size(mDrug)) ones(size(fDrug))];

flatdata = data(:);
groups = {treatment(:);sex(:)};

%1 0 is groups, 0 1 is sex, 1 1 is interaction term
terms = [1 0; 0 1; 1 1];

anovan(flatdata,groups,'model',terms,'varnames',{'treatment','sex'});

end

