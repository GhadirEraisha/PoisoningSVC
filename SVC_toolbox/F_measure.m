function [f_measure, precision, recall, specificity, accuracy]= F_measure(y,score)
            idx0 = (y()==0);
            idx1 = (y()==1);
            idx2 = (y()==2);
            idx3 = (y()==3);
            idx4 = (y()==4);

            p = length(y(~idx0));
            n = length(y(idx0));
            N = p+n;

            tp = sum(y(~idx0)==score(~idx0));
            tn = sum(y(idx0)==score(idx0));
            fp = n-tn;
            fn = p-tp;

            tp_rate = tp/p;
            tn_rate = tn/n;

            accuracy = (tp+tn)/N;
            precision = tp/(tp+fp);
            recall = tp_rate;
            specificity = tn_rate;
            f_measure = 2*((precision*recall)/(precision + recall));
end