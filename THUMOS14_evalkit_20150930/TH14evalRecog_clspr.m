function [ap]=TH14evalRecog_clspr(conf,labels)
%function [rec,prec,ap]=TH14evalRecog_clspr(conf,labels)



% for i=1:101
%     [rec_all(:,i),prec_all(:,i),ap_all(:,i)]=TH14eventclspr(InputScore(:,i),gtlabel(:,i));
% end
% map=mean(ap_all);
% fprintf('\n\nMAP: %f \n\n',map);

% conf: 1500x1 double,   labels: 1574x1 double
% rec: 1500x1   prec: 1500x1   ap: 0.068

[so,sortind]=sort(-conf);
tp=labels(sortind)==1;
fp=labels(sortind)~=1;
npos=length(find(labels==1));

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);

% compute average precision

ap=0;
tmp=labels(sortind)==1;
for i=1:length(conf)
    if tmp(i)==1
        ap=ap+prec(i);
    end
end
ap=ap/npos;