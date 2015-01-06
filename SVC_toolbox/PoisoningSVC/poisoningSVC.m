%1- apply SVC on X
clc; clear;
load ring.mat;
data.X=train_input';
options=struct('method','CG','ker','rbf','arg',0.5,'C',0.1);
[model]=svc(data,options);

%2- find boundaries for SVs, BSVs and Insiders
X_sv = train_input(model.sv_ind,:);
X_bsv = train_input(model.bsv_ind,:);
X_inside = train_input(model.inside_ind,:);

sv_range = [min(X_sv(:,1)),max(X_sv(:,1)),...
    min(X_sv(:,2)),max(X_sv(:,2))];
bsv_range = [min(X_bsv(:,1)),max(X_bsv(:,1)),...
    min(X_bsv(:,2)),max(X_bsv(:,2))];
inside_range = [min(X_inside(:,1)),max(X_inside(:,1)),...
    min(X_inside(:,2)),max(X_inside(:,2))];
X_range = [min(train_input(:,1)),max(train_input(:,1)),...
    min(train_input(:,2)),max(train_input(:,2))];

%3- get the bounding box; a 2d grid enclosing all the points of X
grid_size = 20;
xs = linspace(-3,3,grid_size);
ys = linspace(-3,3,grid_size);
[x, y] = meshgrid(xs,ys);
Z = zeros(size(x));

%4- for each point in the bounding box, compute the f score when it is
%added to the data set
for i=1:size(x,1)
    display(i);
    for j=1:size(x,2)
        p = [x(i,j), y(i,j)];
        poison_data.X = [data.X(:,:), p'];
        [poison_model]=svc(poison_data,options);

        poison_model.cluster_labels = poison_model.cluster_labels(:,1:end-1);
        Z(i,j) = F_measure(model.cluster_labels', poison_model.cluster_labels');

    end
end
[minNum, minInx] = min(Z(:));
[row, col] = ind2sub(size(Z), minInx);
display(minNum);
minPoint = [x(row,col), y(row,col)];
display(minPoint);
figure(1);
[~,h] = contourf(x,y,Z);
h.LineColor = 'none';
colormap(hot); 
colorbar;
hold on;
plot(minPoint(1), minPoint(2), 'yx');
hold on;
plotsvc(data,model);

