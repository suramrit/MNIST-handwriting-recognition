num = xlsread('data.xlsx');
[rows, columns] = size(num);
mcol = 1;
UBitName = ['s' 'u' 'r' 'a' 'm' 'r' 'i' 't'];
personNumber = ['5' '0' '1' '6' '9' '9' '1' '8'];
for col = 3 : 6
    sums = 0;
    for row = 1 : rows
        sums = sums + num(row,col);
    end
    meanCol(mcol) = sums / rows;
    mcol = mcol+1;
end
%variance nd std dev calculation

for col = 3:6
    for row = 1: rows
        vMat(1,row) = num(row,col);
        rvMat(row,col-2) = num(row,col);
    end
       varMat(1,col-2) = var(vMat,0);
       stdDev(1,col-2) = (varMat(1,col-2))^(1/2);
end    

CovMat = cov(rvMat);

for col= 3
    for row = 1:49
        X(row,col-2)= num(row,col);
        Y(row,col-2)= num(row,col+1);
    end
end
covarianceMat = cov(rvMat);
correlationMat= corrcoef(rvMat);
%finding log likelihood of each random variable
variance = var(rvMat);
var1 = variance(1);
var2 = variance(2);
var3 = variance(3);
var4 = variance(4);
mu = mean(rvMat);
mu1 = mu(1);
mu2 = mu(2);
mu3 = mu(3);
mu4 = mu(4);
sigma = sqrt(variance);
sigma1 = sigma(1);
sigma2 = sigma(2);
sigma3 = sigma(3);
sigma4 = sigma(4);

%LogOneVar = sum(log((normpdf(rvMat(:,1),mu(1),sigma(1)))),1);

% find above for each variable and then sum forte the total log likelihood
temp =0 ;
for i = 1:4
LogVar = sum(log((normpdf(rvMat(:,i),mu(i),sigma(i)))),1);
temp = temp + LogVar;
end
logLikelihood = temp;
%creating all possible 4X4 Matrices
matrices = reshape(dec2bin((0:2^16-1),16).'-'0', 4,4,[]);
%accessing a particular 4x4 matrix from the above colection

%convert matrix to sparse matrix
%now check if graph by this var is acyclic
%graphisdag(Sparse Matrix)

%if true then calculate log likelihood for this 
%find highest
%plot BN graphs using -- view(biograph(matrices(:,:,n))) for visualization..

%finding the poition of each diagonal matrics in our exhaustive collection 
n = 1;
j = 0;
i = 1;

while(n ~= 65537)
if graphisdag(sparse(matrices(:,:,n))) == 1;
    j = j+1;
    dag_mat(j,1) = n;
end
n = n+1;
end
n = 1;
while(n ~= 65537)
if graphisdag(sparse(matrices(:,:,n))) == 1
    j = j+1;
    dag_mat(j,1) = n;
end
n = n+1;
end
%selecting a sample Bayesian network from 500+ combinations...
%try to make this generic..
%try to make this generic..
BNgraph = matrices(:,:,10253);
nodes = (matrices(:,:,10253));
log_temp = 0;
log_temp_2 = 0;

i = 1;
j = 1;
%calculate P(1,3)
pmat1 = [];
pmat1(:,1) = rvMat(:,1);
pmat1(:,2) = rvMat(:,3);
pmean1(:,1) = mu(:,1);
pmean1(:,2) = mu(:,3);
pcov1 = covarianceMat([1,3],[1,3]);
like2log1 = sum(log(mvnpdf(pmat1, pmean1, pcov1)));
%calculate P(4,1,2)
pmat2 = [];
pmat2(:,1) = rvMat(:,4);
pmat2(:,2) = rvMat(:,1);
pmat2(:,3) = rvMat(:,2);
pmean2(:,1) = mu(:,4);
pmean2(:,2) = mu(:,1);
pmean2(:,3) = mu(:,2);
pcov2 = covarianceMat([4,(1:2)],[4,(1:2)]);
like2log2 = sum(log(mvnpdf(pmat2, pmean2, pcov2)));
%calculate P(1,2)
pmat3 = [];
pmat3(:,1) = rvMat(:,1);
pmat3(:,2) = rvMat(:,2);
pmean3(:,1) = mu(:,1);
pmean3(:,2) = mu(:,2);
pcov3 = covarianceMat([1,2],[1,2]);
like2log3 = sum(log(mvnpdf(pmat3, pmean3, pcov3)));
%calculate P(2,1)
pmat4 = [];
pmat4(:,1) = rvMat(:,2);
pmat4(:,2) = rvMat(:,1);
pmean4(:,1) = mu(:,2);
pmean4(:,2) = mu(:,1);
pcov4 = covarianceMat([2,1],[2,1]);
like2log4 = sum(log(mvnpdf(pmat4, pmean4, pcov4)));
%calculate P(3)
pmat5 = [];
pmat5(:,1) = rvMat(:,2);
pmat5(:,2) = rvMat(:,1);
pmean5(:,1) = mu(:,2);
pmean5(:,2) = mu(:,1);
pcov5 = covarianceMat(3,3);
log_one = sum(log((mvnpdf(rvMat(:,3),mu(3),pcov5))));
%derived from the formula infered from for the given matrix.... try and
%implement genric code so that you can get optimal value.
BNlogLikelihood = like2log1 + like2log2 - like2log3 +like2log4 + log_one - sum(log((normpdf(rvMat(:,1),mu(1),sigma(1)))),1)- sum(log((normpdf(rvMat(:,3),mu(3),sigma(3)))),1);


