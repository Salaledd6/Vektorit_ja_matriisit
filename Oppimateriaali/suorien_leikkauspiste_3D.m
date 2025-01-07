function [P,Q,t,s]=suorien_leikkauspiste_3D(A,v,C,w)
%P=A+t*v, Q=C+t*w

n=cross(v,w);
vk=cross(n,v);
wk=cross(n,w);
 
t=dot(C-A,wk)/dot(v,wk);
s=dot(A-C,vk)/dot(w,vk);
P=A+t*v;
Q=C+s*w;