import gtsam.*
path = 'C:\Users\a4ama\Desktop\gtsam\parking-garage.g2o';
fileID = fopen(path,'r');

data = textscan(fileID,'%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f');

%Loading data
pose = []; edge = [];
for j=1:size(data{1})
       if data{1}(j)=="VERTEX_SE3:QUAT"
           pose = [pose; data{2}(j) data{3}(j) data{4}(j) data{5}(j) data{6}(j) data{7}(j) data{8}(j) data{9}(j)];
       elseif data{1}(j)=="EDGE_SE3:QUAT"
           edge = [edge; data{2}(j) data{3}(j) data{4}(j) data{5}(j) data{6}(j) data{7}(j) data{8}(j) data{9}(j)...
                    data{10}(j) data{11}(j) data{12}(j) data{13}(j) data{14}(j) data{15}(j) data{16}(j) data{17}(j) data{18}(j)...
                    data{19}(j) data{20}(j) data{21}(j) data{22}(j) data{23}(j) data{24}(j) data{25}(j) data{26}(j)...
                    data{27}(j) data{28}(j) data{29}(j) data{30}(j) data{31}(j)];
       end 
end

isamParams = ISAM2Params;
isam = gtsam.ISAM2(isamParams);

%homogeneous transformation matrix
htm = quat2tform([pose(1,8),pose(1,5),pose(1,6),pose(1,7)]); 

graph = NonlinearFactorGraph;
priorNoise = noiseModel.Diagonal.Sigmas([0.1; 0.1; 0.1; 0.1; 0.1; 0.1]);
graph.add(PriorFactorPose3(0, Pose3(htm), priorNoise));

initials = Values;
initials.insert(pose(1,1), Pose3(htm));
isam.update(graph, initials);
results= isam.calculateEstimate();

for k=1:size(pose,1)-1
    graph = NonlinearFactorGraph;
    initials = Values;
    prevPose = results.at(pose(k,1));
    initials.insert(pose(k+1,1), prevPose); 
    
    for j=1:size(edge,1)
        if edge(j,2) == pose(k+1,1)
            info_matrix = eye(6);
            X = edge(k,25:30);
            info_matrix(4,4) = X(1,1);
            info_matrix(4,5) = X(1,2);
            info_matrix(5,4) = X(1,2);
            info_matrix(4,6) = X(1,3);
            info_matrix(6,4) = X(1,3);
            info_matrix(5,5) = X(1,4);
            info_matrix(5,6) = X(1,5);
            info_matrix(6,5) = X(1,5);
            info_matrix(6,6) = X(1,6);
            covariance = inv(info_matrix);
            Modelcov = noiseModel.Gaussian.Covariance(covariance);
            pose_new = quat2tform([edge(j,9),edge(j,6),edge(j,7),edge(j,8)]);
            pose_new(1:3,4) = edge(j,3:5)';
            graph.add(BetweenFactorPose3(edge(j,1), edge(j,2), Pose3(pose_new), Modelcov));  
            end
    end

    isam.update(graph, initials);
    results= isam.calculateEstimate();
end

final = [];
for i=0:results.size()-2
    final(i+1,1) = results.at(i+1).x;
    final(i+1,2) = results.at(i+1).y;
    final(i+1,3) = results.at(i+1).z;
end

plot3(pose(:,2),pose(:,3),pose(:,4), 'r');
hold on
plot3(final(:,1),final(:,2),final(:,3),'b');
legend("Unoptimized Trajectory","Optimized Trajectory");
axis equal