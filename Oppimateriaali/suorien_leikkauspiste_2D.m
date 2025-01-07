function [P,t,s]=suorien_leikkauspiste_2D(A,v,C,w)
%P=A+t*v=C+s*w
vk=[-v(2),v(1)];
wk=[-w(2),w(1)];
AC=C-A;
t=dot(AC,wk)/dot(v,wk); %jos dot(v,wk)=0, niin suorat samansuuntaiset
CA=-AC;
s=dot(CA,vk)/dot(w,vk);
P=A+t*v