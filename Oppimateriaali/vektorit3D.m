%% s.3-4
clear
B=[600,480,0]
D=[280,0,510]
E=[0,210,400]
BD=D-B
BE=E-B
norm(BD)
norm(BE)

A=[600,0,0]
C=[0,480,0]

plot3([A(1),B(1),C(1)],[A(2),B(2),C(2)],...
      [A(3),B(3),C(3)],'b','linewidth',2)
hold
plot3([D(1),B(1),E(1)],[D(2),B(2),E(2)],[D(3),B(3),E(3)],'r','linewidth',2)
L=800
plot3([0,L],[0,0],[0,0],'k','linewidth',1.5)
plot3([0,0],[0,L],[0,0],'k','linewidth',1.5)
plot3([0,0],[0,0],[0,L],'k','linewidth',1.5)
hold off
grid
axis equal
xlabel('x')
ylabel('y')
zlabel('z','rotation',0)
%% s.5-6, 3D-käsivarsi, suora kinematiikka
clear 
OM=5
MP=3
theta=60
alfa=30
beta=20

Mz=OM*sind(alfa)
ON=OM*cosd(alfa)
Mx=ON*cosd(theta)
My=ON*sind(theta)
Pz=Mz+MP*sind(alfa+beta)
NQ=MP*cosd(alfa+beta)
OQ=ON+NQ
Px=OQ*cosd(theta)
Py=OQ*sind(theta)

L=5
plot3([0,Mx],[0,My],[0,Mz],'r','linewidth',2)
hold
plot3([Mx,Px],[My,Py],[Mz,Pz],'b','linewidth',2)
plot3([0,Px],[0,Py],[0,0],'g')
plot3([Mx,Mx],[My,My],[Mz,0],'k')
plot3([Px,Px],[Py,Py],[Pz,0],'k')
plot3([0,L],[0,0],[0,0],'k','linewidth',1.5)
plot3([0,0],[0,L],[0,0],'k','linewidth',1.5)
plot3([0,0],[0,0],[0,L],'k','linewidth',1.5)
hold off
grid
xlabel('x')
ylabel('y')
zlabel('z','rotation',0)
view([5,3,2])
title({['OM = ',num2str(OM),', MP = ',num2str(MP),', \theta = ',num2str(theta),...
      '^\circ, \alpha = ',num2str(alfa),'^\circ, \beta = ',num2str(beta),'^\circ'];...
      ['P = [',num2str(Px),' ,',num2str(Py),' ,',num2str(Pz),']']})
axis equal

%% s.7--8 käänteinen kinematiikka
Qx=Px
Qy=Py
theta=atan2d(Qy,Qx)
OQ=sqrt(Qx^2+Qy^2)
kPOQ=atand(Pz/OQ)
OP=sqrt(Px^2+Py^2+Pz^2)
kPOM=acosd((OP^2+OM^2-MP^2)/(2*OP*OM))
alfa=kPOQ-kPOM
kOMP=acosd((OM^2+MP^2-OP^2)/(2*OM*MP))
beta=180-kOMP
%% s.13 suoran parametrimuoto
clear
close all
A=[1,2,3]
v=[2,3,-2]

t=1.2 
P=A+t*v
Av=A+1*v 
tmin=-1
tmax=2
Pmin=A+tmin*v
Pmax=A+tmax*v
figure(1) 

plot3([Pmin(1),Pmax(1)],[Pmin(2),Pmax(2)],[Pmin(3),Pmax(3)],'r','linewidth',1.5)
hold
p2=plot3([A(1),Av(1)],[A(2),Av(2)],[A(3),Av(3)],'b','linewidth',3) %vektori v
p1=plot3(A(1),A(2),A(3),'k.','markersize',25) %piste A
p3=plot3(P(1),P(2),P(3),'r.','markersize',20) %piste P
L=5
plot3([-L,L],[0,0],[0,0],'k')
plot3([0,0],[-L,L],[0,0],'k')
plot3([0,0],[0,0],[-L,L],'k')
hold off
grid %taustaristikko
axis equal %akseleiden mittakaavat yhtäsuuriksi
legend([p1,p2,p3],{'A','v','P'},'fontsize',12)
title(['t = ',num2str(t),', P = A+tv'])
xlabel('x')
ylabel('y')

%% s.17 suuntakosinit
clear
v=[1,2,3]
v0=v/norm(v)
alfa=acosd(v0(1))
beta=acosd(v0(2))
gamma=acosd(v0(3))

%%  s. 19
 clear
 close all
 h=4 %pyramidin korkeus
 s=3 %pohjaneliön sivu
 hp=1 %||HP||
 hq=3 %||HQ||
 
 %koordinaatit

 A=[s/2,-s/2,0]
 B=[s/2,s/2,0]
 C=[-s/2,s/2,0]
 D=[-s/2,-s/2,0]
 H=[0,0,h]

 HA=A-H
 HA0=HA/norm(HA)
 HP=hp*HA0
 P=H+HP

 HC=C-H
 HC0=HC/norm(HC)
 HQ=hq*HC0
 Q=H+HQ
 
 PQ=Q-P
 pq=norm(PQ)
 %%

 x1=[A(1),B(1),C(1),D(1),A(1)]
 y1=[A(2),B(2),C(2),D(2),A(2)]
 z1=[A(3),B(3),C(3),D(3),A(3)]
 
 x2=[A(1),H(1),B(1)]
 y2=[A(2),H(2),B(2)]
 z2=[A(3),H(3),B(3)]
  
 x3=[C(1),H(1),D(1)]
 y3=[C(2),H(2),D(2)]
 z3=[C(3),H(3),D(3)]
  
 x4=[P(1),Q(1)]
 y4=[P(2),Q(2)]
 z4=[P(3),Q(3)]
  

 plot3(x1,y1,z1,'k','linewidth',3) 
 hold on
 plot3(x2,y2,z2,'k','linewidth',3) 
 plot3(x3,y3,z3,'k','linewidth',3) 
 plot3(x4,y4,z4,'r','linewidth',2)
 p1=plot3(P(1),P(2),P(3),'m.','markersize',25)
 p2=plot3(Q(1),Q(2),Q(3),'c.','markersize',25)
 L=1.2*s
 plot3([L,0],[0,0],[0,0],'k')
 plot3([0,0],[L,0],[0,0],'k')
 plot3([0,0],[0,0],[0,1.2*h],'k')
 hold off
 %axis([-s,s,-s,s,0,h+1]) %axis([xmin,xmax,ymin,ymax,zmin,zmax])
 axis equal
 grid
 view([1,0.3,0.4]) %katsomis-suunta pisteestä [x,y,z] kohti origoa [0,0,0]
 xlabel('x')
 ylabel('y')
 zlabel('z')
 legend([p1,p2],{'P','Q'},'fontsize',12)
 title({['h = ',num2str(h),', s = ',num2str(s),', hp = ',num2str(hp),...
        ', hq = ',num2str(hq)];['P = [',num2str(P,3),'], Q = [',num2str(Q,3),...
        ']'];['PQ = [',num2str(Q-P,3),'], ||PQ|| = ',num2str(pq,3)]},'fontsize',12)
%% s.21
clear
close all
A=[-10,-20,0]
B=[-35,30,0]
C=[60,25,0]
P=[0,0,65]
F=55


PA=A-P
u=PA/norm(PA)
PB=B-P
v=PB/norm(PB)
PC=C-P
w=PC/norm(PC)
%%
D=u(1)*v(2)*w(3)-u(1)*v(3)*w(2)...
 -u(2)*v(1)*w(3)+u(2)*v(3)*w(1)...
 +u(3)*v(1)*w(2)-u(3)*v(2)*w(1)

%vektoreiden FA,FB,FC pituudet
fa=(v(2)*w(1)-v(1)*w(2))/D*F
fb=(u(1)*w(2)-u(2)*w(1))/D*F
fc=(u(2)*v(1)-u(1)*v(2))/D*F
%%
FA=fa*u
FB=fb*v
FC=fc*w
%%
%tarkastus
FA+FB+FC
[0,0,-F]
%%
plot3([P(1),A(1)],[P(2),A(2)],[P(3),A(3)],'r','linewidth',1.0)
hold
plot3([P(1),B(1)],[P(2),B(2)],[P(3),B(3)],'g','linewidth',1.0)
plot3([P(1),C(1)],[P(2),C(2)],[P(3),C(3)],'b','linewidth',1.0)
L=80
plot3([-L,L],[0,0],[0,0],'k')
plot3([0,0],[-L,L],[0,0],'k')
plot3([0,0],[0,0],[0,L],'k')
plot3(A(1),A(2),A(3),'r.','markersize',20)
plot3(B(1),B(2),B(3),'g.','markersize',20)
plot3(C(1),C(2),C(3),'b.','markersize',20)
PFA=P+FA
p1=plot3([P(1),PFA(1)],[P(2),PFA(2)],[P(3),PFA(3)],'r','linewidth',3)
PFB=P+FB
p2=plot3([P(1),PFB(1)],[P(2),PFB(2)],[P(3),PFB(3)],'g','linewidth',3)
PFC=P+FC
p3=plot3([P(1),PFC(1)],[P(2),PFC(2)],[P(3),PFC(3)],'b','linewidth',3)
p4=plot3([P(1),P(1)],[P(2),P(2)],[P(3),P(3)-F],'k','linewidth',3)
plot3(P(1),P(2),P(3),'k.','markersize',20)
hold off
grid
axis equal
xlabel('x')
ylabel('y')
title({['A = [',num2str(A(1)),',',num2str(A(2)),',',num2str(A(3)),...
               '], B = [',num2str(B(1)),',',num2str(B(2)),',',num2str(B(3)),...
'], C = [',num2str(C(1)),',',num2str(C(2)),',',num2str(C(3)),...
'], P = [',num2str(P(1)),',',num2str(P(2)),',',num2str(P(3)),...
'], F = ',num2str(F)];...
 ['||F_A|| = ',num2str(fa,3),', ||F_B|| = ',num2str(fb,3),...
     ', ||F_C|| = ',num2str(fc,3)]})
legend([p1,p2,p3,p4],{'F_A','F_B','F_C','F'},'fontsize',11)
     
%% s.27, vektoreiden välinen kulma 
clear
u=[1,2,3]
v=[2,-1,5]
alfa=acosd(dot(u,v)/(norm(u)*norm(v)))

%% s.31, komponentteihin jako
clear
close all
v=[1,2,3]
u=[2,-1,5]
vu=dot(v,u)/norm(u)^2*u
vuk=v-vu
vu+vuk
%%
plot3([0,u(1)],[0,u(2)],[0,u(3)],'r','linewidth',2)
hold on
plot3([0,v(1)],[0,v(2)],[0,v(3)],'b','linewidth',2)
plot3([0,vu(1)],[0,vu(2)],[0,vu(3)],'k','linewidth',2)
plot3([vu(1),v(1)],[vu(2),v(2)],[vu(3),v(3)],'m','linewidth',2)

a=7
plot3([-a a],[0 0],[0 0],'k')
plot3([0 0],[-a a],[0 0],'k')
plot3([0 0],[0 0],[0 a],'k')
hold off
grid
axis equal
xlabel('x')
ylabel('y')
legend({'\bf{u}','\bf{v}','\bf{v_u}','\bf{v_{uk}}'},'fontsize',12)
title(['v_u = ',num2str(dot(v,u)/norm(u)^2),' u'],'fontsize',12)

%% s. 39
clear
close all
 
h=4 %pyramidin korkeus
s=2 %pohjaneliön sivu
 
 
%koordinaatit
  
A=[s/2,-s/2,0]
B=[s/2,s/2,0]
C=[-s/2,s/2,0]
D=[-s/2,-s/2,0]
H=[0,0,h]
  
AB=B-A
AH=H-A
BC=C-B
BH=H-B
 
n1=cross(AB,AH)
n2=cross(BC,BH)
v=cross(n1,BH)
w=cross(BH,n2)
%v:n ja w:n välinen kulma
alfa=acosd(dot(v,w)/(norm(v)*norm(w)))
%%
%kuva
x1=[A(1),B(1),C(1),D(1),A(1)]
y1=[A(2),B(2),C(2),D(2),A(2)]
z1=[A(3),B(3),C(3),D(3),A(3)]
  
x2=[A(1),H(1),B(1)]
y2=[A(2),H(2),B(2)]
z2=[A(3),H(3),B(3)]
  
x3=[C(1),H(1),D(1)]
y3=[C(2),H(2),D(2)]
z3=[C(3),H(3),D(3)]
  
 
plot3(x1,y1,z1,'k','linewidth',3) 
hold on
plot3(x2,y2,z2,'k','linewidth',3) 
plot3(x3,y3,z3,'k','linewidth',3)
%sivun BH keskipiste
E=(B+H)/2
%yksikkövektorit
v0=v/norm(v)
w0=w/norm(w)


vx=[E(1),E(1)+v0(1)]
vy=[E(2),E(2)+v0(2)]
vz=[E(3),E(3)+v0(3)]
  
wx=[E(1),E(1)+w0(1)]
wy=[E(2),E(2)+w0(2)]
wz=[E(3),E(3)+w0(3)]
  
plot3(vx,vy,vz,'r','linewidth',4)
plot3(wx,wy,wz,'b','linewidth',4)
   
hold off
axis([-s,s,-s,s,0,h+1]) %axis([xmin,xmax,ymin,ymax,zmin,zmax])
axis equal
grid
view([1,0.3,0.4]) %katsomissuunta pisteestä [x,y,z] kohti origoa [0,0,0]
xlabel('x')
ylabel('y')
zlabel('z')
title(['\alpha = ',num2str(alfa)])

%% s.41, levy, jota taivuttamalla pyramidi syntyy 
HA=A-H
HB=B-H
%HA:n ja HB:n välinen kulma
beta=acosd(dot(HA,HB)/(norm(HA)*norm(HB)))
r=norm(HA)
%levyn nurkkapisteet
n=4
k=0:n
x=r*cosd(k*beta);
y=r*sind(k*beta);

figure(2)
plot(x,y,'k','linewidth',2)
hold
for m=0:n
    plot([0,r*cosd(m*beta)],[0,r*sind(m*beta)],'k','linewidth',2)
end
hold off
grid
axis equal

%% s.43 pyörivä kappale
clear
n=[1,2,3]
omega=2*pi %rad/sek
w=omega*n/norm(n)
A=[5,6,4]
OA=A
v=cross(w,OA)
a=cross(w,v)

OP=dot(OA,n)/norm(n)^2*n
P=OP

plot3([0,n(1)],[0,n(2)],[0,n(3)],'k','linewidth',3)
hold
plot3([0,A(1)],[0,A(2)],[0,A(3)],'b','linewidth',3)
plot3([A(1),P(1)],[A(2),P(2)],[A(3),P(3)],'r','linewidth',2)
v0=v/norm(v)
Av0=A+v0
a0=a/norm(a)
Aa0=A+a0
p1=plot3([A(1),Av0(1)],[A(2),Av0(2)],[A(3),Av0(3)],'m','linewidth',3)
p2=plot3([A(1),Aa0(1)],[A(2),Aa0(2)],[A(3),Aa0(3)],'c','linewidth',3)
plot3(A(1),A(2),A(3),'b.','markersize',30)
plot3([0,P(1)],[0,P(2)],[0,P(3)],'k','linewidth',2)
plot3(P(1),P(2),P(3),'k.','markersize',30)
M=5
plot3([0,M],[0,0],[0,0],'k')
plot3([0,0],[0,M],[0,0],'k')
plot3([0,0],[0,0],[0,M],'k')
hold off
grid
axis equal
xlabel('x')
ylabel('y')
zlabel('z')
legend([p1,p2],{'v^0','a^0'})
title(['||v|| = ',num2str(norm(v),3),', ||a|| = ',num2str(norm(a),3)])
%% s.45, tason normaalimuoto
clear
close all
A=[1,2,3]%tason piste
n=[1,1,2]%tason normaali
a=n(1)
b=n(2)
c=n(3)
d=dot(A,n) %Ax*nx+Ay*ny+Az*nz
  
L=2 %tason koko

%nurkkapisteiden koordinaatit
xn=[A(1)-L,A(1)+L]
yn=[A(2)-L,A(2)+L]
[x,y]=meshgrid(xn,yn) %xy-parit, 
%x = xy-parien x-koordinatit, y = y-koordinaatit
%xn=[-1,3],yn=[0,4]
%x = -1     3
%    -1     3
%y =  0     0
%     4     4
%%
z=1/c*(-a*x-b*y+d) %vastaavat z:n arvot
%%  

surf(x,y,z,'facecolor','c') %surf=surface=pinta, 
hold on
plot3(A(1),A(2),A(3),'r.','markersize',20)
An=A+n
plot3([A(1),An(1)],[A(2),An(2)],[A(3),An(3)],'k','linewidth',2)
hold off
grid on
axis equal
alpha(0.7) %läpinäkyvyys väliltä 0...1, 0=läpinäkyvä, 1 = ei läpinäkyvä
%ei toimi  octavessa 
xlabel('x')
ylabel('y')
title(['A = [',num2str(A),'], n = [',num2str(n),']'])
view([5,-2,2]) %katsomispiste 
%% s.49 tason parametrimuoto 
clear
close all

A=[1,2,3]
u=[2,1,-1]
v=[1,3,1]
 
%parametrien s ja t arvot nurkkapisteille
sn=[-2,3]
tn=[-2,3]
[s,t]=meshgrid(sn,tn) %st-parit
%%
%nurkkapisteiden koordinaatit
x=A(1)+s*u(1)+t*v(1) 
y=A(2)+s*u(2)+t*v(2)
z=A(3)+s*u(3)+t*v(3)
%%
n=cross(u,v) %tason normaali


surf(x,y,z,'facecolor','c')
hold on
p1=plot3(A(1),A(2),A(3),'k.','markersize',20)
Au=A+u
p2=plot3([A(1),Au(1)],[A(2),Au(2)],[A(3),Au(3)],'r','linewidth',3)
Av=A+v
p3=plot3([A(1),Av(1)],[A(2),Av(2)],[A(3),Av(3)],'b','linewidth',3)
An=A+n
p4=plot3([A(1),An(1)],[A(2),An(2)],[A(3),An(3)],'k','linewidth',3)
hold off
grid on
alpha(0.6) %läpinäkyvyys väliltä 0...1, 0=läpinäkyvä, 1 = ei läpinäkyvä, ei octavessa
axis equal
xlabel('x')
ylabel('y')
title(['A= [',num2str(A),'], u = [',num2str(u),'], v = [',num2str(v),'], n = [',...
        num2str(n),']'])
view([5,3,1])
legend([p1,p2,p3,p4],{'A','u','v','n'},'fontsize',12)
%%  s.53 pisteen projektio tasolle 
clear
close all
%pisteiden A,B ja C määräämä taso 
A=[3,2,1]
B=[1,4,1]
C=[-2,1,2]
P=[1,3,8]

AB=B-A
AC=C-A
u=AB
v=AC
n=cross(u,v) %tason normaali
AP=P-A
QP=dot(AP,n)/norm(n)^2*n
AQ=AP-QP
Q=A+AQ

dot(AQ,n)
dot(QP,AQ)
%%

sn=[-2,3];
tn=[-2,3];
[s,t]=meshgrid(sn,tn);
x=A(1)+s*u(1)+t*v(1);
y=A(2)+s*u(2)+t*v(2);
z=A(3)+s*u(3)+t*v(3);

surf(x,y,z,'facecolor','c')
hold on
plot3(A(1),A(2),A(3),'r.','markersize',20)
plot3(P(1),P(2),P(3),'g.','markersize',20)
plot3(Q(1),Q(2),Q(3),'b.','markersize',20)
plot3([A(1),P(1)],[A(2),P(2)],[A(3),P(3)],'g','linewidth',1.5)
plot3([Q(1),P(1)],[Q(2),P(2)],[Q(3),P(3)],'k','linewidth',1.5)
plot3([Q(1),A(1)],[Q(2),A(2)],[Q(3),A(3)],'b','linewidth',1.5)
hold off
grid on
alpha(0.4) %läpinäkyvyys väliltä 0...1, 0=läpinäkyvä, 1 = ei läpinäkyvä
axis equal
xlabel('x')
ylabel('y')


%% s.55, 3D-suorien leikkauspiste
clear
close all
A=[7,1,3]
v=[2,4,0]
C=[1,3,3]
w=[1,-1,0]
n=cross(v,w)
vk=cross(n,v)
wk=cross(n,w)
t=dot(C-A,wk)/dot(v,wk)
s=dot(A-C,vk)/dot(w,vk)
P=A+t*v
Q=C+s*w
%% PQ kohtisuorassa v:n ja w:n kanssa
PQ=Q-P
dot(PQ,v)
dot(PQ,w)
norm(PQ)
%%
%tai suoraan funktiotiedostolla suorien_leikkauspiste_3D.m (oletushakemistossa)
[P,Q,t,s]=suorien_leikkauspiste_3D(A,v,C,w)
%%
p1=plot3(A(1),A(2),A(3),'b.','markersize',25)
hold on
tmin=min([-1,-t])
tmax=max([2,2*t])
Amin=A+tmin*v
Amax=A+tmax*v
plot3([Amin(1),Amax(1)],[Amin(2),Amax(2)],[Amin(3),Amax(3)],'g','linewidth',1.5)  
smin=min([-1,-s])
smax=max([2,2*s])
Cmin=C+smin*w
Cmax=C+smax*w
plot3([Cmin(1),Cmax(1)],[Cmin(2),Cmax(2)],[Cmin(3),Cmax(3)],'r','linewidth',1.5)
Av=A+v
p2=plot3([A(1),Av(1)],[A(2),Av(2)],[A(3),Av(3)],'b','linewidth',3)
p3=plot3(C(1),C(2),C(3),'k.','markersize',25)
Cw=C+w 
p4=plot3([C(1),Cw(1)],[C(2),Cw(2)],[C(3),Cw(3)],'k','linewidth',3)
p5=plot3(P(1),P(2),P(3),'g.','markersize',20)
p6=plot3(Q(1),Q(2),Q(3),'r.','markersize',20)
plot3([P(1),Q(1)],[P(2),Q(2)],[P(3),Q(3)],'c','linewidth',2)
hold off
title({['A = [',num2str(A),'], v = [',num2str(v),'], C = [',num2str(C),...
    '], w = [',num2str(w),']'];['t = ',num2str(t,3),', s = ',num2str(s,3),...
    ', PQ = [',num2str(Q-P,3),']']},'fontsize',9)
    
legend([p1,p2,p3,p4,p5,p6],'A','v','C','w','P','Q')
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')

%% s.61 suoran D,v ja tason A,n leikkauspiste
clear
close all
D=[3,2,7]%suoran piste
v=[1,1,-2]%suoran suuntavektori
A=[1,2,3]%tason piste
n=[-1,-2,5]%tason normaali
AD=D-A
t=-dot(AD,n)/dot(v,n)
P=D+t*v
%testi: AP ja n kohtisuoria
AP=P-A
dot(AP,n)
%% tason yhtälö ax+by+cz=d
a=n(1)
b=n(2)
c=n(3)
d=dot(A,n) %Ax*nx+Ay*ny+Az*nz
  
L=5 %tason koko

%nurkkapisteiden koordinaatit
xn=[A(1)-L,A(1)+L]
yn=[A(2)-L,A(2)+L]
[x,y]=meshgrid(xn,yn) %xy-parit,  
z=1/c*(-a*x-b*y+d) %vastaavat z:n arvot
  %%
surf(x,y,z,'facecolor','c')
hold on
plot3(A(1),A(2),A(3),'b.','markersize',20)
plot3(D(1),D(2),D(3),'k.','markersize',20)
plot3(P(1),P(2),P(3),'r.','markersize',20)
plot3([D(1),P(1)],[D(2),P(2)],[D(3),P(3)],'r','linewidth',1.5)
Dv=D+v
plot3([D(1),Dv(1)],[D(2),Dv(2)],[D(3),Dv(3)],'k','linewidth',3)
hold off
grid on
alpha(0.6) %läpinäkyvyys väliltä 0...1, 0=läpinäkyvä, 1 = ei läpinäkyvä, ei octavessa
axis equal
xlabel('x')
ylabel('y')
title(['t = ',num2str(t)])
%% s.63 tasojen leikkaussuora
clear
A=[3,2,2]
n=[1,0,0.8]
C=[7,5,0]
m=[0,0,1]
%tasojen suuntaiset (yksikkövektorit)
u=cross(n,m) %leikkaussuoran suuntainen
u=u/norm(u)
v=cross(n,u) %tason A,n suuntainen, kohtisuorassa leikkaussuoraa vastaan
v=v/norm(v)
w=cross(m,u) %tason C,m suuntainen, kohtisuorassa leikkaussuoraa vastaan
w=w/norm(w)
%suoran A,v ja tason C,m leikkauspiste
t=-dot(A-C,m)/dot(v,m)
P=A+t*v
%suoran C,w ja tason A,n leikkauspiste
s=-dot(C-A,n)/dot(w,n)
Q=C+s*w

%tason A,n nurkkapisteet
%v:n suuntaan
tmin=min([t,0])-1
tmax=max([t,0])+1
tn1=[tmin,tmax]
%u:n suuntaan
PQ=Q-P
pq=norm(PQ)
u0=PQ/pq %suunta P->Q

smin=-1
smax=pq+1
sn1=[smin,smax]
[t1,s1]=meshgrid(tn1,sn1)
%%
xAn=A(1)+t1*v(1)+s1*u0(1)
yAn=A(2)+t1*v(2)+s1*u0(2)
zAn=A(3)+t1*v(3)+s1*u0(3)


%tason C,m nurkkapisteet
%w:n suuntaan
tmin=min([s,0])-1
tmax=max([s,0])+1
tn2=[tmin,tmax]
%u:n suuntaan
smin=-pq-1
smax=1
sn2=[smin,smax]
[t2,s2]=meshgrid(tn2,sn2)

xCm=C(1)+t2*w(1)+s2*u0(1)
yCm=C(2)+t2*w(2)+s2*u0(2)
zCm=C(3)+t2*w(3)+s2*u0(3)

%leikkaussuoran päätepisteet
Pa=P-u0
Pl=Q+u0

surf(xAn,yAn,zAn,'facecolor','r')
hold
surf(xCm,yCm,zCm,'facecolor','b')
plot3(A(1),A(2),A(3),'r.','markersize',25)
plot3(C(1),C(2),C(3),'b.','markersize',25)
plot3(P(1),P(2),P(3),'m.','markersize',25)
plot3(Q(1),Q(2),Q(3),'c.','markersize',25)

An=A+n
plot3([A(1),An(1)],[A(2),An(2)],[A(3),An(3)],'r','linewidth',3)
Cm=C+m
plot3([C(1),Cm(1)],[C(2),Cm(2)],[C(3),Cm(3)],'b','linewidth',3)
Au=A+u
plot3([A(1),Au(1)],[A(2),Au(2)],[A(3),Au(3)],'k','linewidth',4)
Av=A+v
plot3([A(1),Av(1)],[A(2),Av(2)],[A(3),Av(3)],'m','linewidth',4)
Cw=C+w
plot3([C(1),Cw(1)],[C(2),Cw(2)],[C(3),Cw(3)],'c','linewidth',4)
plot3([Pa(1),Pl(1)],[Pa(2),Pl(2)],[Pa(3),Pl(3)],'k','linewidth',2)
Cu=C+u
plot3([C(1),Cu(1)],[C(2),Cu(2)],[C(3),Cu(3)],'k','linewidth',4)
plot3([P(1),A(1)],[P(2),A(2)],[P(3),A(3)],'m','linewidth',1.5)
plot3([Q(1),C(1)],[Q(2),C(2)],[Q(3),C(3)],'c','linewidth',1.5)
hold off
alpha(0.3)
grid on
axis equal
legend('A,{\bf n}','C,{\bf m}','A','C','P','Q','{\bf n}','{\bf m}','{\bf u}','{\bf v}','{\bf w}')
xlabel('x')
ylabel('y')

%% s.65 3D-ympyrä  

clear
close all
%keskipiste
P=[1,2,3]
%säde 
r=2
%ympyrän tason normaali
n=[3,-1,5]
n=n/norm(n)

%tapa 1
w=[1,1,0] %mikä tahansa vektori, kunhan ei n:n suuntainen
u=cross(n,w) %vektori ympyrän tasossa 
u=u/norm(u)
%tapa 2
%jos A on ympyrän piste
%u=A-P
%u=u/norm(u)

v=cross(n,u) %u:ta vastaan kohtisuora vektori ympyrän tasossa 
v=v/norm(v)

%ympyrän pisteet
t=0:1:360;%kiertokulma, u:sta kohti v:tä
%t=0->P+r*u, t=90->P+r*v  

x=P(1)+r*cosd(t)*u(1)+r*sind(t)*v(1);
y=P(2)+r*cosd(t)*u(2)+r*sind(t)*v(2);
z=P(3)+r*cosd(t)*u(3)+r*sind(t)*v(3);

plot3(P(1),P(2),P(3),'b.','markersize',20)
hold on
Pu=P+u
plot3([P(1),Pu(1)],[P(2),Pu(2)],[P(3),Pu(3)],'r','linewidth',3)
Pv=P+v
plot3([P(1),Pv(1)],[P(2),Pv(2)],[P(3),Pv(3)],'g','linewidth',3)
Pn=P+n
plot3([P(1),Pn(1)],[P(2),Pn(2)],[P(3),Pn(3)],'k','linewidth',3)
plot3(x,y,z,'b','linewidth',2)

hold off
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z')
legend('P','u','v','n')

%% s.69 pallo

clear
%close all
%keskipiste P
x0=0
y0=0
z0=0
P=[x0,y0,z0]
%säde
r=2
%pallokoordinaatti-kulmat
theta=0:10:360;
phi=0:10:180;
%theta-phi-parit
[t,p]=meshgrid(theta,phi)
%%
% vastaavat pallon pisteiden koordinaatit
x=x0+r*sind(p).*cosd(t); 
y=y0+r*sind(p).*sind(t);
z=z0+r*cosd(p);
%%
%leveyspiiri
p0=35
xlev=x0+r*sind(p0).*cosd(theta); 
ylev=y0+r*sind(p0).*sind(theta);
zlev=z0+r*cosd(p0)*ones(1,length(theta)); %ones(1,N) = N:n pituinen vaakavektori ykkösiä

%pituuspiiri
t0=60
xpit=x0+r*sind(phi).*cosd(t0); 
ypit=y0+r*sind(phi).*sind(t0);
zpit=z0+r*cosd(phi);

%pallon piste A, leveyspiiri p0, pituuspiiri t0
Ax=x0+r*sind(p0).*cosd(t0); 
Ay=y0+r*sind(p0).*sind(t0);
Az=z0+r*cosd(p0);
A=[Ax,Ay,Az]

%s.69, tangenttitaso pisteessä A
n=A-P %PA
%suunta A:sta itään
E=cross([0,0,1],n)
E=E/norm(E)
%suunta A:sta pohjoiseen
N=cross(n,E)
N=N/norm(N)
%pallon tangenttitaso
u=E
v=N
sn=[-0.75,0.75]*r;
tn=[-0.75,0.75]*r;
[s,t]=meshgrid(sn,tn);
xtt=A(1)+s*u(1)+t*v(1);
ytt=A(2)+s*u(2)+t*v(2);
ztt=A(3)+s*u(3)+t*v(3);


figure(1)
surf(x,y,z,'facecolor','g','edgecolor','k')
hold on
%plot3(x,y,z,'k.','markersize',12)
surf(xtt,ytt,ztt,'facecolor','c')
plot3(xlev,ylev,zlev,'b','linewidth',2)
plot3(xpit,ypit,zpit,'r','linewidth',2)
p1=plot3(Ax,Ay,Az,'m.','markersize',30)
plot3(x0,y0,z0,'k.','markersize',30)
plot3([x0,Ax],[y0,Ay],[z0,Az],'m','linewidth',2)
plot3([x0,x0+r+1],[y0,y0],[z0,z0],'k','linewidth',2)
plot3([x0,x0],[y0,y0+r+1],[z0,z0],'k','linewidth',2)
plot3([x0,x0],[y0,y0],[z0,z0+r+1],'k','linewidth',2)
AE=A+E
p2=plot3([A(1),AE(1)],[A(2),AE(2)],[A(3),AE(3)],'b','linewidth',3)
AN=A+N
p3=plot3([A(1),AN(1)],[A(2),AN(2)],[A(3),AN(3)],'r','linewidth',3)
hold off
alpha(0.8)
grid on
axis equal
xlabel('x')
ylabel('y')
title(['\theta = ',num2str(t0),'^o, \phi = ',num2str(p0),'^o'])
view([0.8*Ax,0.4*Ay,0.2*Az])
legend([p1,p2,p3],{'A','E','N'})

%% s.73, Pallon ja tason leikkausympyrä
clear 
%taso A,n
A=[3,6,1] %tason piste
n=[1,2,5] %tason normaali
%pallo P,R 
P=[0,0,0] %keskipiste
R=5 %säde

%leikkausympyrä
PA=A-P
PAn=dot(PA,n)/norm(n)^2*n 

%leikkaavat, jos ||PAn||<R

Q=P+PAn %keskipiste
pq=norm(PAn)
r=sqrt(R^2-pq^2) %säde

%leikkausympyrän piste (suunnassa Q->A)
QA=A-Q
B=Q+r*QA/norm(QA)

%tason suuntaiset yksikkövektorit
u=B-Q
u=u/norm(u)

v=cross(n,u)  
v=v/norm(v)


%ympyrän pisteet
t=0:1:360;%kiertokulma, u:sta kohti v:tä
%t=0->Q+r*u, t=90->Q+r*v  

xy=Q(1)+r*cosd(t)*u(1)+r*sind(t)*v(1);
yy=Q(2)+r*cosd(t)*u(2)+r*sind(t)*v(2);
zy=Q(3)+r*cosd(t)*u(3)+r*sind(t)*v(3);

%tason nurkkapisteet

%u:n suunnassa
tmin=-2*r
tmax=norm(QA)+r
tn=[tmin,tmax]
%v:n suunnassa
smin=-2*r
smax=2*r
sn=[smin,smax]
[t,s]=meshgrid(tn,sn)
%nurkkapisteiden koordinaatit
xt=Q(1)+t*u(1)+s*v(1) 
yt=Q(2)+t*u(2)+s*v(2)
zt=Q(3)+t*u(3)+s*v(3)


%pallokoordinaatti-kulmat
theta=0:10:360;
phi=0:10:180;
%theta-phi-parit
[t,p]=meshgrid(theta,phi);
% vastaavat pallon pisteiden koordinaatit
xp=P(1)+R*sind(p).*cosd(t); 
yp=P(2)+R*sind(p).*sind(t);
zp=P(3)+R*cosd(p);

surf(xp,yp,zp,'facecolor','m','edgecolor','none')
hold on
%plot3(xp,yp,zp,'k.','markersize',10)
surf(xt,yt,zt,'facecolor','g')
plot3(xy,yy,zy,'r','linewidth',3)
An=A+n
plot3([A(1),An(1)],[A(2),An(2)],[A(3),An(3)],'k','linewidth',3)
plot3(A(1),A(2),A(3),'b.','markersize',20)
plot3(P(1),P(2),P(3),'m.','markersize',20)
plot3(Q(1),Q(2),Q(3),'r.','markersize',30)
hold off
alpha(0.7)
grid on
axis equal
xlabel('x')
ylabel('y')
view([10,4,3])
lightangle(45,30)
lighting gouraud


%% s.75, lieriö, säde r, korkeus h, akselina z-akseli
r=3
h=7
th=0:10:360;
z=0:h/40:h;
[th,z]=meshgrid(th,z);
x=r*cosd(th);
y=r*sind(th);

figure(1)
surf(x,y,z,'facecolor','g','edgecolor','none')
hold on
%plot3(x,y,z,'k.','markersize',10)
a=2
plot3([-r-a,r+a],[0,0],[0,0],'k','linewidth',2)
plot3([0,0],[-r-a,r+a],[0,0],'k','linewidth',2)
plot3([0,0],[0,0],[0,h+a],'k','linewidth',2)
hold off
grid on
axis equal
%alpha(0.9)
xlabel('x')
ylabel('y')
zlabel('z','rotation',0)
lightangle(45,30)
lighting gouraud
%%
figure(2)
plot(th,z,'k.','markersize',11)
grid
xlim([0,360])
set(gca,'xtick',0:30:360)
xlabel('\theta','fontsize',12)
ylabel('z','rotation',0,'fontsize',12)

%% s.77, lieriön ja tason leikkauskäyrä 
clear
%lieriö
r=3
h=10
th=0:10:360;
z=-h/2:h/20:h/2;
[th,z]=meshgrid(th,z);
x=r*cosd(th);
y=r*sind(th);

%tason piste
A=[0,0,0]
%normaali
ny=-2
nz=5
n=[0,ny,nz] 

%ellipsin puoliakselit
c=-ny/nz*r
a=sqrt(r^2+c^2)
b=r
%yksikkövektorit
u=[0,r/a,c/a]
v=[-1,0,0]
%ellipsin pisteet
kk=0:1:360;
co=cosd(kk);
si=sind(kk);
xe=a*co*u(1)+b*si*v(1);
ye=a*co*u(2)+b*si*v(2);
ze=a*co*u(3)+b*si*v(3);

%polttopisteet
f=sqrt(a^2-b^2)
F1=A+f*u
F2=A-f*u

%puoliakseleiden päätepisteet
pa=A+a*u
pb=A+b*v

%tason nurkkapisteet
tn=[-2*r,2*r]
sn=tn

[t,s]=meshgrid(tn,sn)
xt=A(1)+t*u(1)+s*v(1)
yt=A(2)+t*u(2)+s*v(2)
zt=A(3)+t*u(3)+s*v(3)

figure(1)
surf(x,y,z,'facecolor','g','edgecolor','none')
hold on
surf(xt,yt,zt,'facecolor','c','edgecolor','none')
hold on
plot3(xe,ye,ze,'r','linewidth',5)
% An=A+n
% plot3([A(1),An(1)],[A(2),An(2)],[A(3),An(3)],'k','linewidth',3)
plot3([A(1),pa(1)],[A(2),pa(2)],[A(3),pa(3)],'m','linewidth',3)
plot3([A(1),pb(1)],[A(2),pb(2)],[A(3),pb(3)],'c','linewidth',3)
plot3([F1(1),F2(1)],[F1(2),F2(2)],[F1(3),F2(3)],'r.','markersize',30)
plot3(A(1),A(2),A(3),'b.','markersize',30)
hold off
grid on
axis equal
alpha(0.8)
lightangle(45,30)
lighting gouraud
xlabel('x')
ylabel('y')
zlabel('z','rotation',0)


%% s.79, kartio
clear
R=3 %pohjan säde
h=7 %korkeus

r=0:R/20:R;
th=0:pi/20:2*pi;
[r,th]=meshgrid(r,th);
x=r.*cos(th);
y=r.*sin(th);
z=(1-r/R)*h;

figure(1)
surf(x,y,z,'facecolor','g','edgecolor','none')
hold on
%plot3(x,y,z,'k.','markersize',10)
hold off
grid on
axis equal
xlabel('x')
ylabel('y')
zlabel('z','rotation',0)
alpha(0.8)
lightangle(45,30)
lighting gouraud
%%
figure(2)
subplot(121)
plot(r,th/pi,'ko','markersize',2,'markerfacecolor','k')
grid
xlabel('r')
ylabel('\theta/\pi','rotation',0)
axis square

subplot(122)
plot(x,y,'ko','markersize',2,'markerfacecolor','k')
grid
axis([-R,R,-R,R])
xlabel('x')
ylabel('y','rotation',0)
axis square

%% s.83 torus

clear

R=5
r=2

th=0:10:360;
phi=0:10:360;
[th,phi]=meshgrid(th,phi);
x=(R+r*cosd(phi)).*cosd(th);
y=(R+r*cosd(phi)).*sind(th);
z=r*sind(phi);

surf(x,y,z,'facecolor','g','edgecolor','k')
grid on
axis equal
%alpha(0.9)
xlabel('x')
ylabel('y')
zlabel('z','rotation',0)
lightangle(45,30)
lighting gouraud
title(['R = ',num2str(R),', r = ',num2str(r)])


%% s.85, ellipsoidi
clear
a=5
b=3
c=2

th=0:pi/20:2*pi;
phi=0:pi/20:pi;
[th,phi]=meshgrid(th,phi);
x=a*sin(phi).*cos(th);
y=b*sin(phi).*sin(th);
z=c*cos(phi);

%ellipsit 
kk=0:360;
N=length(kk);
co=cosd(kk);
si=sind(kk);

%z=0->x^2/a^2+y^2/b^2=1
e1x=a*co;
e1y=b*si;
e1z=zeros(1,N);
%y=0->x^2/a^2+z^2/c^2=1
e2x=a*co;
e2y=zeros(1,361);
e2z=c*si;
%x=0 -> y^2/b^2+z^2/c^2=1
e3x=zeros(1,361);
e3y=b*co;
e3z=c*si;


surf(x,y,z,'facecolor','c','edgecolor','none')
hold on
%plot3(x,y,z,'k.','markersize',12)
plot3(e1x,e1y,e1z,'linewidth',2)
plot3(e2x,e2y,e2z,'linewidth',2)
plot3(e3x,e3y,e3z,'linewidth',2)
plot3([0,a],[0,0],[0,0],'r','linewidth',4)
plot3([0,0],[0,b],[0,0],'g','linewidth',4)
plot3([0,0],[0,0],[0,c],'b','linewidth',4)
hold off
grid on
axis equal
alpha(0.5)
lightangle(45,30)
lighting gouraud
xlabel('x','fontsize',13)
ylabel('y','fontsize',13)
zlabel('z','fontsize',13,'rotation',0)

%% s.87 hyperbolinen hyperboloidi
clear
a=2.5
b=2
c=2

th=0:pi/20:2*pi;

%hyperboloidin yläreuna korkeudella z=c*sinh(tmax),
%alareuna korkeudella z=c*sinh(tmin)
%jos tmin=-tmax, niin hyperboloidin korkeus on 2*c*sinh(tmax)
%jos tmax=asinh(1) eli sinh(tmax)=1, niin korkeus on 2*c
tmax=1
tmax=asinh(1)
tmin=-tmax
tt=tmin:(tmax-tmin)/20:tmax;

[th,t]=meshgrid(th,tt);

x=a*cosh(t).*cos(th);
y=b*cosh(t).*sin(th);
z=c*sinh(t);

%z=0->x^2/a^2+y^2/b^2=1, ellipsi
kk=0:360;
N=length(kk)
co=cosd(kk);
si=sind(kk);
e1x=a*co;
e1y=b*si;
e1z=zeros(1,N);

%y=0->x^2/a^2-z^2/c^2=1, hyperbeli
N=length(tt);
h1x=a*cosh(tt);
h1y=zeros(1,N);
h1z=c*sinh(tt);

%x=0->y^2/b^2-z^2/c^2=1, hyperbeli
h2x=zeros(1,N);
h2y=b*cosh(tt);
h2z=c*sinh(tt);




surf(x,y,z,'facecolor','c','edgecolor','none')
hold on
%plot3(x,y,z,'k.','markersize',12)
plot3([0,a],[0,0],[0,0],'r','linewidth',4)
plot3([0,0],[0,b],[0,0],'g','linewidth',4)
plot3([0,0],[0,0],[0,c],'b','linewidth',4)
plot3(e1x,e1y,e1z,'linewidth',2)
plot3(h1x,h1y,h1z,'c','linewidth',2)
plot3(-h1x,h1y,h1z,'c','linewidth',2)
plot3(h2x,h2y,h2z,'m','linewidth',2)
plot3(h2x,-h2y,h2z,'m','linewidth',2)
hold off
axis equal
grid on
alpha(0.7)
lightangle(45,30)
lighting gouraud
view([7,8,4])
xlabel('x','fontsize',13)
ylabel('y','fontsize',13)
zlabel('z','fontsize',13,'rotation',0)




%% s.91, hyperbolinen paraboloidi
clear
a=3
b=2
xmin=-6
xmax=6
xx=xmin:(xmax-xmin)/50:xmax;
ymin=-4
ymax=4
yy=ymin:(ymax-ymin)/50:ymax;
[x,y]=meshgrid(xx,yy);
z=x.^2/a^2-y.^2/b^2;

%z=h>0->x^2/a^2-y^2/b^2=h -> x^2/(sqrt(h)*a)^2-y^2/(sqrt(h)*b)^2=1,   hyperbeli 
h=1
tmin=-1
tmax=1
tt=linspace(tmin,tmax,100);
N=length(tt);
h1x=a*sqrt(h)*cosh(tt);
h1y=b*sqrt(h)*sinh(tt);
h1z=h*ones(1,N);
%z=-h<0->x^2/a^2-y^2/b^2=-h -> -x^2/(sqrt(h)*a)^2+y^2/(sqrt(h)*b)^2=1,   hyperbeli 
h2x=a*sqrt(h)*sinh(tt);
h2y=b*sqrt(h)*cosh(tt);
h2z=-h*ones(1,N);


%y=0->z=x^2/a^2, paraabeli
p1x=xx;
N=length(xx);
p1y=zeros(1,N);
p1z=xx.^2/a^2;

%x=0->z=-y^2/b^2, paraabeli
N=length(yy);
p2x=zeros(1,N);
p2y=yy;
p2z=-yy.^2/b^2;

surf(x,y,z,'facecolor','c','edgecolor','none')
hold on
%plot3(x,y,z,'k.','markersize',10)
plot3(h1x,h1y,h1z,'r','linewidth',2)
plot3(-h1x,h1y,h1z,'r','linewidth',2)
plot3(h2x,h2y,h2z,'g','linewidth',2)
plot3(h2x,-h2y,h2z,'g','linewidth',2)
plot3(p1x,p1y,p1z,'m','linewidth',2)
plot3(p2x,p2y,p2z,'c','linewidth',2)
hold off
axis equal
grid on
alpha(0.8)
lightangle(45,30)
lighting gouraud
view([7,8,4])
xlabel('x','fontsize',13)
ylabel('y','fontsize',13)
zlabel('z','fontsize',13,'rotation',0)

%% s.93 elliptinen paraboloidi
clear
a=3
b=2
xmin=-6
xmax=6
xx=linspace(xmin,xmax,50);
ymin=-4
ymax=4
yy=linspace(ymin,ymax,50);
[x,y]=meshgrid(xx,yy);
z=x.^2/a^2+y.^2/b^2;

%z=h->x^2/a^2+y^2/b^2=h, ellipsi 
kk=0:360;
N=length(kk);
co=cosd(kk);
si=sind(kk);
h=1
e1x=sqrt(h)*a*co;
e1y=sqrt(h)*b*si;
e1z=h*ones(1,N);

%y=0->z=x^2/a^2, paraabeli
p1x=xx;
N=length(xx);
p1y=zeros(1,N);
p1z=xx.^2/a^2;

%x=0->z=y^2/b^2, paraabeli
N=length(yy);
p2x=zeros(1,N);
p2y=yy;
p2z=yy.^2/b^2;

surf(x,y,z,'facecolor','c','edgecolor','none')
hold on
%plot3(x,y,z,'k.','markersize',10)
plot3(e1x,e1y,e1z,'b','linewidth',2)
plot3(p1x,p1y,p1z,'m','linewidth',2)
plot3(p2x,p2y,p2z,'c','linewidth',2)
plot3([0,sqrt(h)*a],[0,0],[h,h],'r','linewidth',4)
plot3([0,0],[0,sqrt(h)*b],[h,h],'g','linewidth',4)
hold off
axis equal
grid on
alpha(0.8)
lightangle(45,30)
lighting gouraud
view([7,8,4])
xlabel('x','fontsize',13)
ylabel('y','fontsize',13)
zlabel('z','fontsize',13,'rotation',0)


%% s.97, hyperbolisen hyperboloidin viivat
clear
close all
a=2
b=2
c=2
th=0:pi/20:2*pi;
tmax=asinh(1)
tmin=-tmax
t=tmin:(tmax-tmin)/30:tmax;
[th,t]=meshgrid(th,t);

x=a*cosh(t).*cos(th);
y=b*cosh(t).*sin(th);
z=c*sinh(t);

%z=0->x^2/a^2+y^2/b^2=1, ellipsi
kk=0:360;
N=length(kk)
co=cosd(kk);
si=sind(kk);
e1x=a*co;
e1y=b*si;
e1z=zeros(1,N);

%z=+-c->x^2/a^2+y^2/b^2=2
%->x^2/(sqrt(2)*a)^2+y^2/(sqrt(2)*b)^2=1, ellipsi
kk=0:360;
N=length(kk)
co=cosd(kk);
si=sind(kk);
e2x=sqrt(2)*a*co;
e2y=sqrt(2)*b*si;
e2z=c*ones(1,N);



surf(x,y,z,'facecolor','c','edgecolor','none')
hold on

%viivat
tt=[-1,1];
for theta=0:20:360
xp=a*cosd(theta)+tt*(-a*sind(theta))
yp=b*sind(theta)+tt*b*cosd(theta)
zp=c*tt
zm=-c*tt
plot3(xp,yp,zp,'c','linewidth',2)
plot3(xp,yp,zm,'m','linewidth',2)
end
plot3(e1x,e1y,e1z,'g','linewidth',2)
plot3(e2x,e2y,e2z,'r','linewidth',2)
plot3(e2x,e2y,-e2z,'b','linewidth',2)
hold off
axis equal
grid on
alpha(0.5)
lightangle(45,30)
lighting gouraud
view([7,8,4])
xlabel('x','fontsize',13)
ylabel('y','fontsize',13)
zlabel('z','fontsize',13,'rotation',0)

%% s.99,  hyperbolisen paraboloidin viivat
clear
a=1.5
b=1.5
xmin=-1
xmax=1
xx=linspace(xmin,xmax,50);
ymin=-1
ymax=1
yy=linspace(ymin,ymax,50);
[x,y]=meshgrid(xx,yy);
z=x.^2/a^2-y.^2/b^2;



surf(x,y,z,'facecolor','c','edgecolor','none')
hold on

%viivat
%x=a*(s+t),y=b*(s-t),z=4*s*t
%-> s=(a*y+b*x)/(2*a*b),t=1/2*(x/a-y/b)
%x=xmin,y=0->s=b*xmin/(2*a*b)=1/2*xmin/a,t=1/2*xmin/a
%x=xmax,y=0->s=b*xmax/(2*a*b)=1/2*xmax/a,t=1/2*xmax/a
%x=0,y=ymin->s=a*ymin/(2*a*b)=1/2*ymin/b,t=-1/2*ymin/b
%x=0,y=ymax->s=a*ymax/(2*a*b)=1/2*ymax/b,t=-1/2*ymax/b


%1) x=xmin,y=0
smin=1/2*xmin/a
tmin=1/2*xmin/a
%2) x=xmax,y=0
smax=1/2*xmax/a
tmax=1/2*xmax/a



%viivat t=vakio
s=[smin,smax]
for t=tmin:(tmax-tmin)/10:tmax;
    xt=a*(s+t);
    yt=b*(s-t);
    zt=4*s*t;
    plot3(xt,yt,zt,'m','linewidth',2)
end

%viivat s=vakio
t=[tmin,tmax]
for s=smin:(smax-smin)/10:smax;
    xt=a*(s+t);
    yt=b*(s-t);
    zt=4*s*t;
    plot3(xt,yt,zt,'c','linewidth',2)
end
hold off
axis equal
grid on
alpha(0.8)
lightangle(45,30)
lighting gouraud
view([7,8,4])
xlabel('x','fontsize',13)
ylabel('y','fontsize',13)
zlabel('z','fontsize',13,'rotation',0)

