%initialize parameters...
%before using labels... remeber that the data format needs to be changed...
clear;
images = loadMNISTImages('train-images');
labels = loadMNISTLabels('train-labels');
label_mat = zeros(60000,10);
for i=1:60000
    val = labels(i);
    label_mat(i,val+1)=1;
end

for i=1:10
    for j= 1:784
        r = 2*rand(1,784);
        weigh(i,:)= r;
    end
end

r = rand(10,1);
b = r;
    
%calculating evidence values... 

for img=1:60000
    for clas =1:10
            a_k(img,clas)= weigh(clas,:)*images(:,img)+b(clas,1);
    end
end
%a_j calculation
a_j = zeros(60000,1);

for img=1:60000
    for clas =1:10
        a_j(img)= a_j(img)+exp(a_k(img,clas)); %check calculation
    end
end
    
%softmax calculation-- evaluating Y        

for img =1:60000
    for clas =1:10
        p_clas_x(img,clas)=  (exp(a_k(img,clas)) / (a_j(img)));
    end
end

%implement mini batch SGD...

%calculate cross entrop error function

error_x = zeros(60000,1);

%for img = 1:60000
 %   for clas = 1:10
  %error_x(img)= -((clas-1)*log(p_clas_x(img,clas)))+error_x(img);
   % end
%end

%gradient of error function
%Yj estimation.. this is basically the p_clas_x value for a particular
%class and image

%del_e_j
%del_e_j = zeros(784,10,60000); %takin only 60000 images intitally
%for img = 1:60000 
 %   for clas =1:10
  %      del_e_j(:,clas,img) = (p_clas_x(img,clas)-label_mat(img,clas))*(images(:,img));
  %  end
%end
ind =1;
sum_del_e_j = zeros(784,10);
%appying m-batch gradient descent
w_old =weigh;
w_updated = zeros(10,784);
eta = 2.6;
batch =380;
a_j_new = zeros(380,1);
for pass = 1:10
    batch =1;
   while batch<55620 
      a_k_new = zeros(10,10);
      a_j_new = zeros(380,1);
      for ind = 1 : 380
        %find sum_del_e_j
      %%fprintf('\nIn image..... %d In Batch %d In Pass..... %d\n Eta val..... %f',ind+batch-1,batch,pass,eta);
      for clas =1:10
            a_k_new(ind,clas)= w_old(clas,:)*images(:,ind+batch-1)+b(clas,1);
            if(isnan(a_k_new(ind,clas)))
            {
                exit;    
            };
            end       
      end
      end
      for ind = 1 : 380
      for clas =1:10
          %fprintf('\n1 In image..... %d \n' ,ind+batch-1);
            a_j_new(ind)= a_j_new(ind)+exp(a_k_new(ind,clas)); %check calculation
           %fprintf('e^a_k val...%f \na_j_new val... %f\n',a_j_new(ind),a_k_new(ind,clas));
      end
      end
      for ind = 1 : 380
      for clas =1:10
           %fprintf('\n2 In image..... %d \n' ,ind+batch-1);
           p_clas_x_new(ind,clas)=  (exp(a_k_new(ind,clas)) / (a_j_new(ind)));
      end
      end
      for ind = 1 : 380
      for clas = 1:10
          %fprintf('\n3 In image..... %d \n' ,ind+batch-1);
        del_e_j(:,clas,ind) = (p_clas_x_new(ind,clas)-label_mat(ind+batch-1,clas))*(images(:,ind+batch-1));
      end
      end
      for ind = 1 :380
      for clas =1:10
          %fprintf('\n4 In image..... %d \n' ,ind+batch-1);
        sum_del_e_j(:,clas) = del_e_j(:,clas,ind)+sum_del_e_j(:,clas); 
      end
      end
            %weight updation for the batch...
     
      for clas = 1:10 
       w_updated(clas,:) = w_old(clas,:) - (eta/380)* (sum_del_e_j(:,clas).');
       %fprintf('----In batch---..... %d\n',batch);
       if(isnan(w_updated(clas,:)))
            {
                exit;    
            };
       end
      end
     %proceed to convergence
      w_old = w_updated;
      batch = batch +380;
      eta = eta *(0.85);
   end
  % ;
  %fprintf('Eta Val: %f',eta)
end
Wlr = w_updated.';
blr = b.';


%%Using single layer neural network.....


imgs = 60000;

train_x = loadMNISTImages('train-images')';
train_x = train_x(1:imgs,:);
train_y = loadMNISTLabels('train-labels');
train_y = train_y(1:imgs,:);

 label_nn = zeros(size(train_y,1),10);
    for i = 1:size(train_y,1)
        label_nn(i,train_y(i)+1) = 1;
    end
train_y=label_nn;


[N, D] = size(train_x);

pass = 1;

eta = 4.7;

hid_unit =500;

K = 10;


w = -2+(4)*rand(hid_unit,D ); 

v = -2+(4)*rand(K, hid_unit);

batch_num = 1000;

batch_count = floor(N/batch_num);

batch_ind = reshape([1:batch_num*batch_count]',batch_num, batch_count);
batch_ind = batch_ind';

if N - batch_num*batch_count >0
    batch_ind(end+1,:)=batch_ind(end,:);
    batch_ind(end,1:(N - batch_num*batch_count)) = [batch_num*batch_count+1: N];
end

for epoch = 1:pass
    for batch = 1:batch_count
        X=[];
        Y=[];
        for j=1:batch_num
            X=[X; train_x(batch_ind(batch,j),:)];
            Y=[Y; train_y(batch_ind(batch,j),:)];
        end
       
        z = X*w';
     
        z = tanh(z);

        ydash = z*v';
      
        for i = 1:size(ydash,1)
            deno =  sum(exp(ydash(i,:)));
            ydash(i,:) = exp(ydash(i,:))/deno;
        end
        
        hid_unit=size(z,2)-1;
        N=size(X,1);

        deltav = (Y-ydash)'*z;
   
        z=1-z.^2;

        deltaw = ((Y-ydash)*v).*z;
      
        deltaw = deltaw'*X;

        deltav=deltav./N;
        deltaw=deltaw./N;


        w = w + eta.*deltaw;
        v = v + (eta).*deltav;
    end
 
    z=[];
    
    z = train_x*w';
    
    z = tanh(z);
  
    ydash = z*v';
    
    for i = 1:size(ydash,1)
        deno =  sum(exp(ydash(i,:)));
        ydash(i,:) = exp(ydash(i,:))/deno;
    end
    z=[];
    for i=1:size(ydash,1)
    [ u vv ]= max(ydash(i,:));
    ydash(i,:)=0;
    ydash(i,vv)=1;
    end
    ydash=[];
    z=[];
end



h='tanh';
bnn1=zeros(1,500);
bnn2=zeros(1,K);
Wnn1=w.';
Wnn2=v.';






