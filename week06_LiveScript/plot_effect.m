function plot_effect(mCtrl, fCtrl, mDrug, fDrug)

figure;
hold on;

errorbar(1,mean(mCtrl), std(mCtrl)/sqrt(numel(mCtrl)),'k');
errorbar(2,mean(fCtrl), std(fCtrl)/sqrt(numel(fCtrl)),'k');

errorbar(1,mean(mDrug), std(mDrug)/sqrt(numel(mDrug)),'r');
errorbar(2,mean(fDrug), std(fDrug)/sqrt(numel(fDrug)),'r');

plot([1 2], [mean(mCtrl) mean(fCtrl)], 'k');
plot([1 2], [mean(mDrug) mean(fDrug)], 'r');

ylabel('Locomotor activity (m)');
xlabel('Sex');

xlim([0 3]);

end

