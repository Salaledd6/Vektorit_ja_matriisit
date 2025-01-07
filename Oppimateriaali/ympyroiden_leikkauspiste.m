function [lp1,lp2]=ympyroiden_leikkauspiste(P1,P2,r1,r2)
%P1=[x1,y1] ympyran 1 kp, säde r1
%P2=[x2,y2] ympyran 2 kp, säde r2
%lp1=[lp1x,lp1y],lp2=[lp2x,lp2y] niiden leikkauspisteet 
%kts. ympyroiden_leikkauspiste.pdf
x1=P1(1);
y1=P1(2);
x2=P2(1);
y2=P2(2);
if abs(y1-y2)>0.01
A=-2*(x2-x1)/(2*(y2-y1));
B=(r1^2-r2^2-(x1^2-x2^2)-(y1^2-y2^2))/(2*(y2-y1));
a=1+A^2;
b=-2*x1+2*A*B-2*y1*A;
c=x1^2+B^2-2*y1*B+y1^2-r1^2;
lp1x=(-b-sqrt(b^2-4*a*c))/(2*a);
lp2x=(-b+sqrt(b^2-4*a*c))/(2*a);
lp1y=A*lp1x+B;
lp2y=A*lp2x+B;
else
    lp1x=(r1^2-r2^2-(x1^2-x2^2))/(2*(x2-x1));
    lp2x=lp1x;
    x=lp1x;
    a=1
    b=-2*y1
    c=x^2-2*x1*x+x1^2+y1^2-r1^2
    d=b^2-4*a*c
    lp1y=(-b-sqrt(b^2-4*a*c))/(2*a)
    lp2y=(-b+sqrt(b^2-4*a*c))/(2*a)
end
lp1=[lp1x,lp1y];
lp2=[lp2x,lp2y];

