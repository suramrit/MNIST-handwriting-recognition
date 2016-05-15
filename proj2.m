%The code uses a .xlsx file which contains the real world data.
%.xlsx file was generating the real world data txt file in excel and
%extracting the values.

real_world = xlsread('input real world.xlsx');

for i = 1:69623
    label(i) = real_world(i,1);
end

for j = 1:69623
    col = 1;
    for k = 2:47
        feature_matrix(j,col) = real_world(j,k);
        col=col+1;
     end
end

%partioning

for j = 1:55698
    for k = 1:46
        training_set_matrix(j,k) = feature_matrix(j,k);
     end
end

for j = 55699:62661
    for k = 1:46
     l = j-55698;
     validation_set_matrix(l,k) = feature_matrix(j,k);
    end
end

for j = 62662:69623
    for k = 1:46
     l = j-62661;
     test_set_matrix(l,k) = feature_matrix(j,k);
    end
end

%part2
for j = 1:55698
        training_set_label(j) = label(j);
     
end

for j = 55699:62661
     l = j-55698;
     validation_set_label(l) = label(j);
end

for j = 62662:69623
     l = j-62661;
     test_set_label(l) = label(j);
end


%setting trainInd1
for i = 1:55698
    trainInd1(i,1) = i;
end
%setting validInd1
ind = 1;
for i = 55699:62611
    validInd1(ind,1) = i;
    ind=ind+1;
end

lambda1=1;
%calculating basis functions
%choosing M
M1 = 6;
%calculating phi-j---------------------------------
data_points=[156 2351 3431 15644 30009 52344];
%storing mu1:
for i = 1:46
    for j = 1:M1
        mu1(i,j)=training_set_matrix(data_points(j),i);
    end
end
%variance calculation:
for i = 1:46
    %for j = 1:55698
     %var_cal(j)=training_set_matrix(j,i);
    %end
    var_value = var(training_set_matrix(:,i));
    sigma(i,i)=(var_value); %using sigma prop to ith dim var(x)
end
sigma;

%setting----Sigma1
for i = 1:M1
    Sigma1(:,:,i)=sigma;
end

for i = 1:M1
    for j= 1:46
        for k=1:46
            if(j~=k)
                
                    Sigma1(j,k,i)=.3;
            end
        end
    end
end
cond(Sigma1(:,:,1))
Sigma1_inv = Sigma1(:,:,1)\eye(46,46);
%currently using sigma-j as sigma
%calculating phi (design matrix)
for i = 1:55698
    for j= 1:M1
       if(j==1)
        design_mat(i,j) = 1;
       else
                                              %removed one transpose from
       diff = transpose(training_set_matrix(i,1:46))-mu1(:,j);                                       %here 
       design_mat(i,j)=exp((-0.5)*(transpose(diff))*(Sigma1_inv)*diff);
       end
    end
end
%calculating w-ml--- from training data
%take lambda = 0.75

w_ml(:,:) = (inv(1*eye(M1,M1)+(transpose(design_mat)*design_mat)))*(transpose(design_mat)*transpose(training_set_label));
w1=w_ml; %correct till w1

%---------------calculating Ed(w) for training set------------------
negate=transpose(w_ml)*transpose(design_mat);
E_dw =0;
for i = 1:55698
   E_dw = E_dw + ((training_set_label(i)-(transpose(w_ml)*transpose(design_mat(i,:))))^2);
end
%calculating E-nw
mod_sum=0;
for i = 1:M1
    pow = (w_ml(i))*(w_ml(i));
    mod_w(i) = abs(pow);
    mod_sum = mod_sum+abs(pow);
end
mod_sum = mod_sum *(0.75/2);
%finding normalised error---training data
E_norm_training = mod_sum + (1/2)*(E_dw) ;
%finding E-rms training data
E_rms_training = sqrt(E_dw/55698);

% -------------------calculating E-rms for validation set--------------
% create design matrix for validation set----
%variance calculation:


%currently using sigma-j as sigma
for i = 1:6913
    for j= 1:M1
       if(j==1)
        design_mat_validation(i,j) = 1;
       else
       diff = transpose(validation_set_matrix(i,1:46))-mu1(:,j);                                       %here 
       design_mat_validation(i,j)=exp((-0.5)*(transpose(diff))*(Sigma1_inv)*diff);
        %design_mat_validation(i,j)=exp((-0.5)*(transpose(transpose(validation_set_matrix(i,1:46))-mu1(:,j)))*(sigma2)*(transpose(validation_set_matrix(i,1:46))-mu1(:,j)));
       end
    end
end


%---------------calculating Ed(w) for validation set------------------
negate=transpose(w_ml)*transpose(design_mat_validation);
E_dw_valid =0;
for i = 1:6913
   E_dw_valid = E_dw_valid + ((validation_set_label(i)-(transpose(w_ml)*transpose(design_mat_validation(i,:))))^2);
end

%finding normalised error---validation data
E_norm_validation = mod_sum + (1/2)*(E_dw_valid) ;
%finding E-rms validation data
E_rms_validation = sqrt(E_dw_valid/6913);
validPer1 = E_rms_validation;
%--------synthetic-------------------%
%partitioning
for i = 1:1600
    trainInd2(i,1) = i;
end
%setting validInd1
ind = 1;
for i = 1601:2000
    validInd2(ind,1) = i;
    ind=ind+1;
end
min_rms =10000000000000;
lfinal=0;
mfinal=0;
y=transpose(x);
count =0;
for j = 1:1600
    for k = 1:10
        synthetic_training_matrix(j,k) = y(j,k);
     end
end

for j = 1601:2000
    for k = 1:10
     l_val = j-1600;
     synthetic_validation_matrix(l_val,k) = y(j,k);
    end
end

for j = 1:1600
        synthetic_training_label(j) = t(j);
     
end

for j = 1601:2000
     l_val = j-1600;
     synthetic_validation_label(l_val) = t(j);
end

for i = 1:10
    %for j = 1:55698
     %var_cal(j)=training_set_matrix(j,i);
    %end
    var_value = var(synthetic_training_matrix(:,i));
    sigma_syn(i,i)=(var_value); %using sigma prop to ith dim var(x)
end


for i = 1:10
    for j = 1:10
        if(i~=j)
            sigma_syn(i,j)=0.3;   %might be making difference.. 
        end
    end
end


Sigma2_inv = sigma_syn\eye(10,10);


for m_val=25:35
    syn_design_matrix=ones(1600,m_val);
    datapoints=randi([1 1600],1,m_val); %change from m-1 
    %display(mu1)
    for j=1:1600
        for k=2:m_val
            syn_design_matrix(j,k)=exp(-0.5*((((synthetic_training_matrix(j,:)-synthetic_training_matrix(datapoints(k-1),:)))*(Sigma2_inv)*((transpose(synthetic_training_matrix(j,:)-synthetic_training_matrix(datapoints(k-1),:)))))));
        end
    end
    %made phi
    for l_val=10:-0.5:1
        count = count+1;
        lamda=eye(m_val)*l_val;
        calc_label = (inv(lamda+(transpose(syn_design_matrix)*syn_design_matrix)))*(transpose(syn_design_matrix)*transpose(synthetic_training_label));
    
        error_dw=0;
        for i = 1:1600
          error_dw = error_dw + ((synthetic_training_label(i)-(transpose(calc_label)*transpose(syn_design_matrix(i,:))))^2);
        end
        %sqared_error=(transpose(synthetic_training_label) - (transpose(weights)*transpose(design_matrix))).^2;
        train_erms_syn = sqrt((error_dw)/1600);
        if(min_rms>train_erms_syn)
           min_rms=train_erms_syn;
           lfinal=l_val;
           mfinal=m_val;
           min_wt = calc_label;
           mu2_min = datapoints;
        end
          plot(m_val,train_erms_syn);
           hold on;
      %display(train_erms)
    end
end
M2=mfinal-1;
%after seeing the Mvs rms plot-- over several iterations.. 
for i=1:M2
    Sigma2(:,:,i)=sigma_syn;
end
lambda2 = lfinal;
%after finding minmum m and lambda giving min rms........................
%storing mu1:

datapoints_valid= randi([1 1600],1,M2);
mu2=zeros(10,M2);

for i = 1:10
    for j = 1:M2
        mu2(i,j)=synthetic_training_matrix(datapoints_valid(j),i);
    end
end

design_mat_valid=zeros(1600,M2);

for i = 1:1600
    for j= 1:M2
       if(j==1)
        design_mat_valid(i,j) = 1;
       else
                                              %removed one transpose from
       diff = transpose(synthetic_training_matrix(i,1:10))-mu2(:,j);%here --- check for dimention comptibility  
       design_mat_valid(i,j)=exp((-0.5)*(transpose(diff))*(Sigma2_inv)*diff);
       end
    end
end

w2= (inv(l_val*eye(M2,M2)+(transpose(design_mat_valid)*design_mat_valid)))*(transpose(design_mat_valid)*transpose(synthetic_training_label));
%---------------calculating Ed(w) for training set------------------
%negate=transpose(w2)*transpose(design_mat);
E_dw =0;
for i = 1:1600
   E_dw = E_dw + ((synthetic_training_label(i)-(transpose(w2)*transpose(design_mat_valid(i,:))))^2);
end

%finding normalised error---training data
trainPer2 = sqrt(E_dw/1600);

% -------------------calculating E-rms for validation set--------------
% create design matrix for validation set----
%variance calculation:

syn_design_mat_validation=zeros(400,M2);
%currently using sigma-j as sigma
for i = 1:400
    for j= 1:M2
       if(j==1)
        syn_design_mat_validation(i,j) = 1;
       else
       diff = transpose(synthetic_validation_matrix(i,1:10))-mu2(:,j);                                       %here 
       syn_design_mat_validation(i,j)=exp((-0.5)*(transpose(diff))*(Sigma2_inv)*diff);
        %design_mat_validation(i,j)=exp((-0.5)*(transpose(transpose(validation_set_matrix(i,1:46))-mu1(:,j)))*(sigma2)*(transpose(validation_set_matrix(i,1:46))-mu1(:,j)));
       end
    end
end


%---------------calculating Ed(w) for validation set------------------
E_dw_valid =0;
for i = 1:400
   E_dw_valid = E_dw_valid + ((synthetic_validation_label(i)-(transpose(w2)*transpose(syn_design_mat_validation(i,:))))^2);
end
E_rms_validation = sqrt(E_dw_valid/400);
validPer2 = E_rms_validation;

%validper2 vals: 
%
%  val     m
%.1536----40
%.15511---20
%.15391---30
%.15405---40
%.15494---50
%.15895---200
%val is min in range of m[25,35]



                
                
                
        








