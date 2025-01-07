function [lp1,lp2]=pallojen_leikkauspiste(P1,P2,P3,r1,r2,r3)
%P1=[x1,y1,z1] pallon 1 kp, säde r1
%P2=[x2,y2,z2] pallon 2 kp, säde r2
%P3=[x3,y3,z3] pallon 3 kp, säde r3
%lp1=[lp1x,lp1y,lp1z],lp2=[lp2x,lp2y,lp2z] niiden leikkauspisteet
%kts. pallojen_leikkauspiste.pdf
x1=P1(1);
y1=P1(2);
z1=P1(3);
x2=P2(1);
y2=P2(2);
z2=P2(3);
x3=P3(1);
y3=P3(2);
z3=P3(3);
A=2*(x2-x1);
B=2*(y2-y1);
E=-2*(z2-z1);
F=r1^2-r2^2-(x1^2-x2^2)-(y1^2-y2^2)-(z1^2-z2^2);
C=2*(x3-x1);
D=2*(y3-y1);
G=-2*(z3-z1);
H=r1^2-r3^2-(x1^2-x3^2)-(y1^2-y3^2)-(z1^2-z3^2);

I=(B*G-D*E)/(B*C-A*D);
J=(H*B-F*D)/(B*C-A*D);
K=(C*E-A*G)/(B*C-A*D);
L=(F*C-A*H)/(B*C-A*D);

a=I^2+K^2+1;
b=2*I*J-2*x1*I+2*K*L-2*y1*K-2*z1;
c=J^2-2*x1*J+x1^2+L^2-2*y1*L+y1^2+z1^2-r1^2;

z0=(-b-sqrt(b^2-4*a*c))/(2*a);
lp1x=I*z0+J;
lp1y=K*z0+L;
lp1z=z0;
lp1=[lp1x,lp1y,lp1z];

z0=(-b+sqrt(b^2-4*a*c))/(2*a);
lp2x=I*z0+J;
lp2y=K*z0+L;
lp2z=z0;
lp2=[lp2x,lp2y,lp2z];
