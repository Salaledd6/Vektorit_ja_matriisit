%%  s.5-6
clear
close all
%kuvion pisteet
P=[1,1
   1,4
   3,4
   3,3
   2,3
   2,1
   1,1]'

%kierto
th=50
c=cosd(th)
s=sind(th)
K=[c -s
   s c]
KP=K*P %kierretty kuvio 


plot(P(1,:),P(2,:),'r','linewidth',2)
hold
plot(KP(1,:),KP(2,:),'b','linewidth',2)
plot(0,0,'g.','markersize',20)
hold off
grid
title(['\theta = ',num2str(th),'^o'])
axis([-3,4,-1,6])
axis square
legend('P','KP')



%%  s.7-8
clear
%close all
m=25
N=2*m %pisteiden lukumäärä
x=3+4*rand(1,N); %rand(1,N)=1xN-matriisi satunnaislukuja väliltä 0..1
y=1+2*rand(1,N);
P=[x;y];
 
v=[1,2] %suoran suunta
  
th=atan2d(v(2),v(1)) %v:n suuntakulma  
c=cosd(th)
s=sind(th)
K=[c -s
   s  c]
KP=K^-1*P; %-th:n verran kierretyt pisteet

xK=KP(1,:);
yK=KP(2,:);
  
[ys,ind]=sort(yK); 
% vektori ys = yK:t suuruusjärjestyksessä, pienin->suurin 
% vektori ind = järjestysnumerot vastaavassa järjestyksessä

y0=(ys(m)+ys(m+1))/2 %vaakasuoran suoran korkeus

xKv=min(xK)-1 %päätepisteiden x-koordinaatit, v=vasen, o = oikea
xKo=max(xK)+1

vaakasuora=[xKv,xKo
             y0,y0]
  
%kierretään päätepisteitä th:n verran
suora_v=K*vaakasuora %v:n suuntaisen suoran päätepisteet
  
%suoran alapuolella olevat pisteet 
xala=x(ind(1:m)); 
yala=y(ind(1:m)); 
%yläpuolella olevat
xyla=x(ind(m+1:N)); 
yyla=y(ind(m+1:N));

%vaakasuoran alapuolella olevat kierretyt pisteet
xKala=xK(ind(1:m)); 
yKala=yK(ind(1:m)); 
%yläpuolella olevat
xKyla=xK(ind(m+1:end));
yKyla=yK(ind(m+1:end));


subplot(1,2,1)
plot(suora_v(1,:),suora_v(2,:),'k','linewidth',2)
hold on
plot([0,v(1)],[0,v(2)],'g','linewidth',3)
plot(0,0,'g.','markersize',20)
plot(xala,yala,'r.',xyla,yyla,'b.','markersize',15)
hold off
grid
axis equal
title(['v = [',num2str(v),'], \theta = ',num2str(th,4),'^o'])




subplot(1,2,2)
plot([xKv,xKo],[y0,y0],'k','linewidth',2)
hold on
plot([0,norm(v)],[0,0],'g','linewidth',3)
plot(0,0,'g.','markersize',20)
plot(xKala,yKala,'r.',xKyla,yKyla,'b.','markersize',15)
hold off
grid
axis equal


%% s. 11-12, kierto pisteen P0 ympäri homogeenisilla koordinaateilla
clear
close all
%P0
x0=4
y0=4
%kiertokulma 
th=50
c=cosd(th)
s=sind(th)
K=[c,-s,0
   s,c,0
   0,0,1]
S=[1,0,x0
   0,1,y0
   0,0,1]
KP0=S*K*S^-1
%kuvion pisteet
F=[0,0
   0.5,0
   0.5,1.5
   1.5,1.5
   1.5,2
   0.5,2
   0.5,2.5
   2,2.5
   2,3
   0,3
   0,0]';

N=length(F(1,:))
%homogeeniset koordinaatit
F=[F;ones(1,N)];
%kierretty kuvio
KP0F=KP0*F;

plot(F(1,:),F(2,:),'b','linewidth',1.5)
hold
plot(KP0F(1,:),KP0F(2,:),'r','linewidth',1.5)
plot(x0,y0,'k.','markersize',20)
hold off
grid
axis equal
legend('F','KF','P_0')
title(['\theta = ',num2str(th),'^o'])

%% s.17- , koordinaatiston muunnos
clear
close all
%uv-koordinaatiston origo P0
x0=2
y0=1
%uv-koordinaatiston kiertokulma
th=30 
c=cosd(th)
s=sind(th)
K=[c,-s,x0
   s,c,y0
   0,0,1]
%uv->xy
%P:n uv-koordinaatit
Pu=-4
Pv=3 
Puv=[Pu;Pv;1]
Pxy=K*Puv %xy-koordinaatit

% % %xy->uv
% Px=-4
% Py=3
% Pxy=[Px;Py;1]
% Puv=K^-1*Pxy

ux=c;
uy=s;
vx=-s;
vy=c;


L=6 %akseleiden pituus
plot([-L L],[0 0],'k','linewidth',2)
hold on
plot([0 0],[-L L],'k','linewidth',2)
plot([x0-L*ux,x0+L*ux],[y0-L*uy,y0+L*uy],'b','linewidth',2)
plot([x0-L*vx,x0+L*vx],[y0-L*vy,y0+L*vy],'b','linewidth',2)
p1=plot([x0,x0+ux],[y0,y0+uy],'r','linewidth',3)
p2=plot([x0,x0+vx],[y0,y0+vy],'g','linewidth',3)
p3=plot(x0,y0,'b.','markersize',20)
plot([x0+Puv(1)*ux,Pxy(1)],[y0+Puv(1)*uy,Pxy(2)],'g')
plot([x0+Puv(2)*vx,Pxy(1)],[y0+Puv(2)*vy,Pxy(2)],'r')
plot([Pxy(1),Pxy(1)],[0,Pxy(2)],'k')
plot([0,Pxy(1)],[Pxy(2),Pxy(2)],'k')
p4=plot(Pxy(1),Pxy(2),'r.','markersize',20)
hold off
grid
axis([-L L -L L])
axis square
title(['P_0 = [',num2str(x0),',',num2str(y0),'], \theta = ',num2str(th),'^o, Pu = ',num2str(Puv(1)),...
     ', Pv = ',num2str(Puv(2)),', Px = ',num2str(Pxy(1)),', Py = ',num2str(Pxy(2))])
legend([p1,p2,p3,p4],{'u','v','P_0','P'},'fontsize',12) 
set(gca,'xtick',-L:L)
set(gca,'ytick',-L:L)


%% s.25, kierretyn  ellipsin yhtälö
clear
%close all
%puoliakselit
a=3 
b=2
%keskipiste
x0=4 
y0=2
%kiertokulma 
th=30 

c=cosd(th)
s=sind(th)
M=[b^2,0,0
   0,a^2,0
   0,0,0]

K=[c,-s,x0
   s,c,y0
   0,0,1]

%kierretyn ellipsin yhtälö Ax^2+Bxy+Cy^2+Dx+Ey=a^2b^2-F
KMK=(K^-1)'*M*K^-1
A=KMK(1,1)
B=2*KMK(1,2)
C=KMK(2,2)
D=2*KMK(1,3)
E=2*KMK(2,3)
F=KMK(3,3)

%ellipsin pisteet
t=0:360;
N=length(t)
%uv-koordinaatit
ue=a*cosd(t);
ve=b*sind(t);
%xy-koordinaatit
xye=K*[ue;ve;ones(1,N)];
xe=xye(1,:);
ye=xye(2,:);

r=5 %u- ja v-akseleiden pituus = 2r
u=[c;s];
v=[-s;c];
plot(xe,ye,'b','linewidth',1.5)
hold 
p1=plot([x0-r*u(1),x0+r*u(1)],[y0-r*u(2),y0+r*u(2)],'r','linewidth',1.5)
plot([x0,x0+u(1)],[y0,y0+u(2)],'r','linewidth',3)
p2=plot([x0-r*v(1),x0+r*v(1)],[y0-r*v(2),y0+r*v(2)],'g','linewidth',1.5)
plot([x0,x0+v(1)],[y0,y0+v(2)],'g','linewidth',3)
plot([-r,r],[0,0],'k')
plot([0,0],[-r,r],'k')
p3=plot(x0,y0,'k.','markersize',15)
hold off
grid
axis equal
title({['\theta = ',num2str(th),...
        '^\circ,  a = ',num2str(a),', b = ',num2str(b),...
        ', x_0 = ',num2str(x0),', y_0 = ',num2str(y0)];...
        [' A  = ',num2str(A,3),...
         ',  B = ',num2str(B,3),', C = ',num2str(C,3),...
         ', D = ',num2str(D,3),', E = ',num2str(E,3),', F = ',num2str(F,3)]},...
         'fontsize',12)
legend([p1,p2,p3],{'u','v','P_0'},...
    'fontsize',12,'location','northwest')

%tarkastus
k=17
x=xe(k)
y=ye(k)
A*x^2+B*x*y+C*y^2+D*x+E*y
a^2*b^2-F

%% s.29 kierretyn hyperbelin yhtälö
clear
%close all
%hyperbelin puoliakselit
a=3
b=2
%keskipiste
x0=4
y0=2
%kiertokulma
th=30 

c=cosd(th)
s=sind(th)
M=[b^2,0,0
   0,-a^2,0
   0,0,0]

K=[c,-s,x0
   s,c,y0
   0,0,1]

%kierretyn ellipsin yhtälö Ax^2+Bxy+Cy^2+Dx+Ey=a^2b^2-F
KMK=(K^-1)'*M*K^-1
A=KMK(1,1)
B=2*KMK(1,2)
C=KMK(2,2)
D=2*KMK(1,3)
E=2*KMK(2,3)
F=KMK(3,3)
%hyperbelin pisteet uv-koordinaatistossa
tmax=1.8
t=-tmax:0.01:tmax;
N=length(t)
uh1=a*cosh(t);
uh2=-a*cosh(t);
vh=b*sinh(t);
%xy-koordinaatistossa
xyh1=K*[uh1;vh;ones(1,N)];
xh1=xyh1(1,:);
yh1=xyh1(2,:);
xyh2=K*[uh2;vh;ones(1,N)];
xh2=xyh2(1,:);
yh2=xyh2(2,:);


r=8 %u- ja v-akseleiden pituus = 2r
u=[c;s];
v=[-s;c];
plot(xh1,yh1,'b',xh2,yh2,'b','linewidth',1.5)
hold 
p1=plot([x0-r*u(1),x0+r*u(1)],[y0-r*u(2),y0+r*u(2)],'r','linewidth',1.5)
plot([x0,x0+u(1)],[y0,y0+u(2)],'r','linewidth',3)
p2=plot([x0-r*v(1),x0+r*v(1)],[y0-r*v(2),y0+r*v(2)],'g','linewidth',1.5)
plot([x0,x0+v(1)],[y0,y0+v(2)],'g','linewidth',3)
plot([-r,r],[0,0],'k')
plot([0,0],[-r,r],'k')
p3=plot(x0,y0,'k.','markersize',20)
hold off
grid
axis equal
title({['\theta = ',num2str(th),...
        '^\circ,  a = ',num2str(a),', b = ',num2str(b),...
        ', x_0 = ',num2str(x0),', y_0 = ',num2str(y0)];...
        [' A  = ',num2str(A,3),...
         ',  B = ',num2str(B,3),', C = ',num2str(C,3),...
         ', D = ',num2str(D,3),', E = ',num2str(E,3),', F = ',num2str(F,3)]},...
         'fontsize',12)
legend([p1,p2,p3],{'u','v','P_0'},...
    'fontsize',12,'location','northwest')


%tarkastus
k=30
x=xh1(k)
y=yh1(k)
A*x^2+B*x*y+C*y^2+D*x+E*y
a^2*b^2-F
%% s.33 3D delta-robotti
clear
%mitat
f=40
e=10
rf=20
re=50

%suora kinematiikka
%varsien F1J1, F2J2 ja F3J3 kulmat
th1=15
th2=-37
th3=-5


F1=[0,-f/2*tand(30),0];
F1J1=[0,-rf*cosd(th1),rf*sind(th1)];
J1=F1+F1J1;

K=[cosd(120),-sind(120)
   sind(120),cosd(120)];

F1xy=[F1(1);F1(2)];
F2xy=K*F1xy;
F2=[F2xy(1),F2xy(2),0];
F2J2xy=K*[0;-rf*cosd(th2)];
F2J2z=rf*sind(th2);
F2J2=[F2J2xy(1),F2J2xy(2),F2J2z];
J2=F2+F2J2;

F3xy=K^-1*F1xy;
F3=[F3xy(1),F3xy(2),0];
F3J3xy=K^-1*[0;-rf*cosd(th3)];
F3J3z=rf*sind(th3);
F3J3=[F3J3xy(1),F3J3xy(2),F3J3z];
J3=F3+F3J3;


E1E0=[0,e/2*tand(30),0];
J1p=J1+E1E0; %J1'

E1E0xy=[E1E0(1);E1E0(2)];

E2E0xy=K*E1E0xy;
E2E0=[E2E0xy(1),E2E0xy(2),0];
J2p=J2+E2E0;%J2'

E3E0xy=K^-1*E1E0xy;
E3E0=[E3E0xy(1),E3E0xy(2),0];
J3p=J3+E3E0;%J3'

%funktio-tiedosto pallojen_leikkauspiste.m oletushakemistossa
[lp1,lp2]=pallojen_leikkauspiste(J1p,J2p,J3p,re,re,re);

if lp1(3)<lp2(3)
    E0=lp1;
else
    E0=lp2;
end



%yläkolmion nurkkapisteet
ylax=[-f/2,f/2,0,-f/2];
a=f/2*tand(30);
b=f/2*tand(60)-a; %yläkolmion korkeus = f/2*tand(60)
ylay=[-a,-a,b,-a];
ylaz=[0,0,0,0];
%alakolmion nurkkapisteet
alax=[E0(1)-e/2,E0(1)+e/2,E0(1),E0(1)-e/2];
a=e/2*tand(30);
b=e/2*tand(60)-a; %alakolmion korkeus = e/2*tand(60)
alay=[E0(2)-a,E0(2)-a,E0(2)+b,E0(2)-a];
alaz=[E0(3),E0(3),E0(3),E0(3)];

E1=E0-E1E0;
E2=E0-E2E0;
E3=E0-E3E0;
plot3(ylax,ylay,ylaz,'b','linewidth',3)
hold
plot3(0,0,0,'b.','markersize',20)
plot3(alax,alay,alaz,'r','linewidth',3)
plot3(E0(1),E0(2),E0(3),'r.','markersize',20)
p1=plot3([F1(1),J1(1)],[F1(2),J1(2)],[F1(3),J1(3)],'c.-','linewidth',3,'markersize',20)
p2=plot3([F2(1),J2(1)],[F2(2),J2(2)],[F2(3),J2(3)],'m.-','linewidth',3,'markersize',20)
p3=plot3([F3(1),J3(1)],[F3(2),J3(2)],[F3(3),J3(3)],'g.-','linewidth',3,'markersize',20)
plot3([J1(1),E1(1)],[J1(2),E1(2)],[J1(3),E1(3)],'k.-','linewidth',3,'markersize',20)
plot3([J2(1),E2(1)],[J2(2),E2(2)],[J2(3),E2(3)],'k.-','linewidth',3,'markersize',20)
plot3([J3(1),E3(1)],[J3(2),E3(2)],[J3(3),E3(3)],'k.-','linewidth',3,'markersize',20)
grid
axis equal
hold off
xlabel('x')
ylabel('y')
title(['\theta_1 = ',num2str(th1),', \theta_2 = ',num2str(th2),...
     ', \theta_3 = ',num2str(th3),',   E_0 = [',num2str(E0),']'],'fontsize',12)
legend([p1,p2,p3],{'F_1J_1','F_2J_2','F_3J_3'},'fontsize',12)



%% käänteinen kinematiikka
% E0=[10,15,-45];
x0=E0(1);
y0=E0(2);
z0=E0(3);

F1=[0,-f/2*tand(30),0];
E1p=[0,y0-e/2*tand(30),z0]; %E1'

%funktio-tiedosto ympyroiden_leikkauspiste.m oletushakemistossa
[lp1,lp2]=ympyroiden_leikkauspiste([F1(2),F1(3)],[E1p(2),E1p(3)],rf,sqrt(re^2-x0^2));

if lp1(1)<lp2(1) 
    J1=lp1;
else
    J1=lp2;
end

th1=atan2d(J1(2)-F1(3),-(J1(1)-F1(2)))


x0y0p=K^-1*[x0;y0]; %[x0';y0']
x0p=x0y0p(1);
y0p=x0y0p(2);

E1p=[0,y0p-e/2*tand(30),z0]; %E1'

[lp1,lp2]=ympyroiden_leikkauspiste([F1(2),F1(3)],[E1p(2),E1p(3)],rf,sqrt(re^2-x0p^2));

if lp1(1)<lp2(1) 
    J2p=lp1
else
    J2p=lp2
end

th2=atan2d(J2p(2)-F1(3),-(J2p(1)-F1(2)))




x0y0pp=K*[x0;y0]; %[x0'';y0'']
x0pp=x0y0pp(1);
y0pp=x0y0pp(2);

E1pp=[0,y0pp-e/2*tand(30),z0]; %E1''


[lp1,lp2]=ympyroiden_leikkauspiste([F1(2),F1(3)],[E1pp(2),E1pp(3)],rf,sqrt(re^2-x0pp^2))

if lp1(1)<lp2(1) 
    J3pp=lp1;
else
    J3pp=lp2;
end

th3=atan2d(J3pp(2)-F1(3),-(J3pp(1)-F1(2)))



%% s.49, 3D-kiertomatriisi
clear
close all
n=[1;1;1]
n=n/norm(n) %kiertoakseli
th=45 %kiertokulma

I3=[1 0 0
    0 1 0
    0 0 1]
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0]

K=cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A

P=[1;2;3] %piste
KP=K*P %kierretty piste
%%
Q=dot(P,n)*n %kiertoympyrän keskipiste kierron akselilla
r=norm(P-Q) %kiertoympyrän säde
%yksikkövektorit ympyrän pisteiden laskemiseksi
u=(P-Q)/r
v=cross(n,u)
v=v/norm(v)
t=0:th/100:th;
yx=Q(1)+0.2*r*cosd(t)*u(1)+0.2*r*sind(t)*v(1);
yy=Q(2)+0.2*r*cosd(t)*u(2)+0.2*r*sind(t)*v(2);
yz=Q(3)+0.2*r*cosd(t)*u(3)+0.2*r*sind(t)*v(3);

t=0:360;
Yx=Q(1)+r*cosd(t)*u(1)+r*sind(t)*v(1);
Yy=Q(2)+r*cosd(t)*u(2)+r*sind(t)*v(2);
Yz=Q(3)+r*cosd(t)*u(3)+r*sind(t)*v(3);


plot3(P(1),P(2),P(3),'b.','markersize',20)
hold on
plot3(KP(1),KP(2),KP(3),'r.','markersize',20)
plot3(Q(1),Q(2),Q(3),'k.','markersize',20)
plot3([0 1.2*Q(1)],[0,1.2*Q(2)],[0,1.2*Q(3)],'k','linewidth',2)
plot3([P(1),Q(1)],[P(2),Q(2)],[P(3),Q(3)],'b','linewidth',2)
plot3([KP(1),Q(1)],[KP(2),Q(2)],[KP(3),Q(3)],'r','linewidth',2)
plot3(yx,yy,yz,'k')
plot3(Yx,Yy,Yz,'k')
plot3([0,norm(P)],[0,0],[0,0],'k','linewidth',1)
plot3([0,0],[0,norm(P)],[0,0],'k','linewidth',1)
plot3([0,0],[0,0],[0,norm(P)],'k','linewidth',1)
hold off
grid on
axis equal
legend({'P','KP','Q','n'},'fontsize',12)
xlabel('x')
view([4,2,1])
ylabel('y')
%% s.53, kierto joka vie yksikkövektorin u yksikkövektoriksi v
clear
close all
u=[1;1;1]
u=u/norm(u)
v=[-1;2;3]
v=v/norm(v)
 
n=cross(u,v)
n=n/norm(n) %akseli
 
th=acosd(dot(u,v)) %u:n ja v:n välinen kulma 

%kiertomatriisi
I3=[1 0 0
    0 1 0
    0 0 1]
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0]

K=cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A

%tarkastus
K*u
v
%% animaatio kierrosta

t=0:0.02:1; %väliasennon ut kiertokulma t*th 

N=length(t)
for k=1:N
   thk=t(k)*th;
   I3=[1 0 0
    0 1 0
    0 0 1];
   A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];

   Kt=cosd(thk)*I3+(1-cosd(thk))*n*n'+sind(thk)*A;

   ut=Kt*u; %u:n väliasento, t(k)*th:n verran kierretty u
   plot3([0 u(1)],[0 u(2)],[0 u(3)],'r','linewidth',2)
   hold on
   plot3([0 v(1)],[0 v(2)],[0 v(3)],'b','linewidth',2)
   plot3([0 n(1)],[0 n(2)],[0 n(3)],'g','linewidth',2)
   plot3([0 ut(1)],[0 ut(2)],[0 ut(3)],'m','linewidth',2)
   hold off
   grid on
   axis([-1,1,-1,1,-1,1])
   axis square
   xlabel('x')
   ylabel('y')
   view([-1 -1 1])
   pause(0.05)
end

%% s.55
clear
close all
m=25
N=2*m
x=1+rand(1,N);
y=2+2*rand(1,N);
z=3+3*rand(1,N);

P=[x;y;z];
 
u=[1;2;3] %tason normaali
u=u/norm(u);
v=[0;0;1];
%kiertomatriisi u->v
n=cross(u,v);
n=n/norm(n);
th=acosd(dot(u,v))

I3=[1 0 0
    0 1 0
    0 0 1]
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0]

K=cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A

KP=K*P; %kierretyt pisteet
xK=KP(1,:);
yK=KP(2,:);
zK=KP(3,:);
  
[zs,ind]=sort(zK); 
% vektori zs = K:t suuruusjärjestyksessä, pienin->suurin 
% vektori ind = järjestysnumerot vastaavassa järjestyksessä

z0=(zs(m)+zs(m+1))/2 %vaakasuoran tason korkeus

xKv=min(xK)-1 %tason nurkkien x-koordinaatit, v=vasen, o = oikea
xKo=max(xK)+1
yKv=min(yK)-1 %tason nurkkien y-koordinaatit, v=vasen, o = oikea
yKo=max(yK)+1

%vaakasuoran tason nurkkapisteet
vaakasuora=[xKv,xKo,xKo,xKv,xKv
            yKv,yKv,yKo,yKo,yKv
            z0,z0,  z0, z0,z0]


%keskipiste
AK=[(xKv+xKo)/2;(yKv+yKo)/2;z0];

%u:ta vastaan kohtisuoran tason nurkkapisteet
%kierretään nurkkapisteitä th:n verran
taso_u=K^-1*vaakasuora 
%u:ta vastaan kohtisuoran tason keskipiste
%kierretään keskipistettä th:n verran
A=K^-1*AK;

%tason alapuolella olevat pisteet 
xala=x(ind(1:m)); 
yala=y(ind(1:m));
zala=z(ind(1:m));
%yläpuolella olevat
xyla=x(ind(m+1:N)); 
yyla=y(ind(m+1:N));
zyla=z(ind(m+1:N));
%vaakasuoran tason alapuolella olevat kierretyt pisteet
xKala=xK(ind(1:m)); 
yKala=yK(ind(1:m));
zKala=zK(ind(1:m));
%yläpuolella olevat
xKyla=xK(ind(m+1:end)); 
yKyla=yK(ind(m+1:end));
zKyla=zK(ind(m+1:end));

figure(1)
%plot3(taso_u(1,:),taso_u(2,:),taso_u(3,:),'k','linewidth',2)
fill3(taso_u(1,:),taso_u(2,:),taso_u(3,:),'c','linewidth',2)
hold on
plot3([A(1),A(1)+u(1)],[A(2),A(2)+u(2)],[A(3),A(3)+u(3)],'g','linewidth',3)
plot3(xala,yala,zala,'b.',xyla,yyla,zyla,'r.','markersize',15)
hold off
grid
axis equal
alpha(0.4)

figure(2)
%plot3(vaakasuora(1,:),vaakasuora(2,:),vaakasuora(3,:),'k','linewidth',2)
fill3(vaakasuora(1,:),vaakasuora(2,:),vaakasuora(3,:),'c','linewidth',2)
hold on
plot3([AK(1),AK(1)+v(1)],[AK(2),AK(2)+v(2)],[AK(3),AK(3)+v(3)],'k','linewidth',3)
plot3(xKala,yKala,zKala,'b.',xKyla,yKyla,zKyla,'r.','markersize',15)
hold off
grid
axis equal
alpha(0.4)

%% s.57, kiertomatriisi K -> akseli n ja kulma th
clear
close all
%kiertomatriisi
K=[0 1  0
   0 0 -1
  -1 0  0]
%akseli ja kulma
th=acosd((K(1,1)+K(2,2)+K(3,3)-1)/2)
nx=(K(3,2)-K(2,3))/(2*sind(th))
ny=(K(1,3)-K(3,1))/(2*sind(th))
nz=(K(2,1)-K(1,2))/(2*sind(th))

n=[nx;ny;nz]
   
%tarkastus
I3=[1 0 0
    0 1 0
    0 0 1]
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0]

cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A
K
%% animaatio kierrosta
t=0:0.01:1;

for k=1:length(t)
%k=1
Kt=cosd(t(k)*th)*I3+(1-cosd(t(k)*th))*n*n'+sind(t(k)*th)*A;
ut=Kt(:,1);
vt=Kt(:,2);
wt=Kt(:,3);
plot3([0,ut(1)],[0,ut(2)],[0,ut(3)],'r','linewidth',2)
hold
plot3([0,vt(1)],[0,vt(2)],[0,vt(3)],'g','linewidth',2)
plot3([0,wt(1)],[0,wt(2)],[0,wt(3)],'b','linewidth',2)
plot3([0,n(1)],[0,n(2)],[0,n(3)],'k','linewidth',2)
plot3([0,1],[0,0],[0,0],'r')
plot3([0,0],[0,1],[0,0],'g')
plot3([0,0],[0,0],[0,1],'b')
hold off
grid
axis([-2,2,-2,2,-2,2])
axis square
view([3,1,0.7])
%xlabel('x')
%ylabel('y')
pause(0.05)
end

%% s.61, kierto [1;0;0]->u, [0;1;0]->v, [0;0;1]->w 
clear
close all
u=[1;2;3]
v=[2;-1;0] %kohtisuora u:n kanssa
w=cross(u,v)
%yksikkövektorit
u0=u/norm(u)
v0=v/norm(v)
w0=w/norm(w)
%kiertomatriisi
K=[u0,v0,w0]
%akseli ja kulma
th=acosd((K(1,1)+K(2,2)+K(3,3)-1)/2)
nx=(K(3,2)-K(2,3))/(2*sind(th))
ny=(K(1,3)-K(3,1))/(2*sind(th))
nz=(K(2,1)-K(1,2))/(2*sind(th))

n=[nx;ny;nz]
   
%tarkastus
I3=[1 0 0
    0 1 0
    0 0 1]
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0]

cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A
K
%% animaatio kierrosta
t=0:0.01:1;

for k=1:length(t)
%k=1
Kt=cosd(t(k)*th)*I3+(1-cosd(t(k)*th))*n*n'+sind(t(k)*th)*A;
ut=Kt(:,1);
vt=Kt(:,2);
wt=Kt(:,3);
plot3([0,ut(1)],[0,ut(2)],[0,ut(3)],'r','linewidth',2)
hold
plot3([0,vt(1)],[0,vt(2)],[0,vt(3)],'g','linewidth',2)
plot3([0,wt(1)],[0,wt(2)],[0,wt(3)],'b','linewidth',2)
plot3([0,n(1)],[0,n(2)],[0,n(3)],'k','linewidth',2)
plot3([0,u0(1)],[0,u0(2)],[0,u0(3)],'r','linewidth',1)
plot3([0,v0(1)],[0,v0(2)],[0,v0(3)],'g','linewidth',1)
plot3([0,w0(1)],[0,w0(2)],[0,w0(3)],'b','linewidth',1)
hold off
grid
axis([-2,2,-2,2,-2,2])
axis square
view([3,1,0.7])
%xlabel('x')
%ylabel('y')
pause(0.05)
end

%% s.63 Eulerin kulmat
clear
close all
%muodostetaan kiertomatriisi K
u=[5;2;2]
v=[-2;4;1] %kohtisuora u:n kanssa
w=cross(u,v)
u0=u/norm(u)
v0=v/norm(v)
w0=w/norm(w)
K=[u0,v0,w0]
%% zxz Eulerin kulmat
alfa=atan2d(K(1,3),-K(2,3))
beta=acosd(K(3,3))
gamma=atan2d(K(3,1),K(3,2))
%%
%K1=K_z,alfa
n=[0;0;1]
th=alfa
I3=[1 0 0
    0 1 0
    0 0 1];
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];    
K1=cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A;
x1=K1*[1;0;0]
y1=K1*[0;1;0]
z1=n
%K2=K_x1,beta
n=x1
th=beta
I3=[1 0 0
    0 1 0
    0 0 1];
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];    
K2=cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A;

x2=x1
y2=K2*y1
z2=K2*z1


%K3=K_z2,gamma
n=z2
th=gamma
I3=[1 0 0
    0 1 0
    0 0 1];
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];    
K3=cosd(th)*I3+(1-cosd(th))*n*n'+sind(th)*A;

x3=K3*x2
y3=K3*y2
z3=z2

%tarkastus
K3*K2*K1
K


%% animaatio

t=0:0.01:1;
N=length(t)


I3=[1 0 0
    0 1 0
    0 0 1];

for k=1:3*N
    
if k<=N
n=[0;0;1];
th=alfa;
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];    
K1=cosd(t(k)*th)*I3+(1-cosd(t(k)*th))*n*n'+sind(t(k)*th)*A;
ut=K1(:,1);
vt=K1(:,2);
wt=n;
elseif k<=2*N
n=x1;
th=beta;
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];    
K2=cosd(t(k-N)*th)*I3+(1-cosd(t(k-N)*th))*n*n'+sind(t(k-N)*th)*A;
ut=n;
vt=K2*y1;
wt=K2*z1; 
else
n=z2;
th=gamma;
A=[0 -n(3) n(2)
   n(3) 0 -n(1)
  -n(2) n(1) 0];    
K3=cosd(t(k-2*N)*th)*I3+(1-cosd(t(k-2*N)*th))*n*n'+sind(t(k-2*N)*th)*A;
ut=K3*x2;
vt=K3*y2;
wt=n;   
end
    
plot3([1,0],[0,0],[0,0],'r')
hold; 
plot3([0,0],[0,1],[0,0],'g')
plot3([0,0],[0,0],[0,1],'b')
plot3([0,ut(1)],[0,ut(2)],[0,ut(3)],'r','linewidth',2)
plot3([0,vt(1)],[0,vt(2)],[0,vt(3)],'g','linewidth',2)
plot3([0,wt(1)],[0,wt(2)],[0,wt(3)],'b','linewidth',2)
plot3([0,u(1)],[0,u(2)],[0,u(3)],'m','linewidth',2)
plot3([0,v(1)],[0,v(2)],[0,v(3)],'color',[0.4660 0.6740 0.1880],'linewidth',2)
plot3([0,w(1)],[0,w(2)],[0,w(3)],'c','linewidth',2)
hold off;
grid
axis([-1,1,-1,1,-1,1])
axis square
view([3,2,1])
title(['\alpha = ',num2str(alfa,3),', \beta = ',num2str(beta,3),...
        ', \gamma = ',num2str(gamma,3)])
pause(0.01)
end
%% s.69 kvaterniot
clear
n=[1,1,1]%kiertoakseli
n=n/norm(n)
th=45*pi/180%kiertokulma
qv=sin(th/2)*n %vektoriosa
q0=cos(th/2) %skalaariosa
Q=[q0,qv] %yksikkökvaternio
Qinv=[q0,-qv] %käänteiskvaternio
P=[1,2,3]%kierrettävä piste
%kertolasku funktiolla quat_prod.m (oletushakemistossa)
QP=quat_prod(Q,[0,P])
QPQinv=quat_prod(QP,Qinv)
KP=QPQinv(2:4)%kierretty piste
%% s.73, koordinaatiston muunnos
clear
close all
P0=[0.5;1;0.5]
u=[1;1;0]
u=u/norm(u)
v=[-1;1;1]
v=v/norm(v)
w=[1;-1;2]
w=w/norm(w)

K=[u,v,w,P0;0,0,0,1]

%uvw -> xyz
Puvw=[1.2;-0.4;0.6;1]
Pxyz=K*Puvw

%xyz -> uvw 
% Pxyz=[2;2;2;1]
% Puvw=K^-1*Pxyz

plot3(P0(1),P0(2),P0(3),'k.','markersize',20)
hold
plot3(Pxyz(1),Pxyz(2),Pxyz(3),'m.','markersize',20)
plot3([P0(1),P0(1)+u(1)],[P0(2),P0(2)+u(2)],[P0(3),P0(3)+u(3)],'r','linewidth',3)
plot3([P0(1),P0(1)+v(1)],[P0(2),P0(2)+v(2)],[P0(3),P0(3)+v(3)],'g','linewidth',3)
plot3([P0(1),P0(1)+w(1)],[P0(2),P0(2)+w(2)],[P0(3),P0(3)+w(3)],'b','linewidth',3)
plot3([P0(1),P0(1)+Puvw(1)*u(1)],[P0(2),P0(2)+Puvw(1)*u(2)],...
    [P0(3),P0(3)+Puvw(1)*u(3)],'r','linewidth',1)
plot3([P0(1)+Puvw(1)*u(1),P0(1)+Puvw(1)*u(1)+Puvw(2)*v(1)],...
      [P0(2)+Puvw(1)*u(2),P0(2)+Puvw(1)*u(2)+Puvw(2)*v(2)],...
      [P0(3)+Puvw(1)*u(3),P0(3)+Puvw(1)*u(3)+Puvw(2)*v(3)],'g','linewidth',1)
plot3([P0(1)+Puvw(1)*u(1)+Puvw(2)*v(1),Pxyz(1)],...
      [P0(2)+Puvw(1)*u(2)+Puvw(2)*v(2),Pxyz(2)],...
      [P0(3)+Puvw(1)*u(3)+Puvw(2)*v(3),Pxyz(3)],'b','linewidth',1)
plot3([0,1],[0,0],[0,0],'k','linewidth',2)
plot3([0,0],[0,1],[0,0],'k','linewidth',2)
plot3([0,0],[0,0],[0,1],'k','linewidth',2)
hold off
grid on
axis equal
view([1,1,0.5])
legend({'P_0','P','u','v','w'},'fontsize',12)
xlabel('x')
ylabel('y')
zlabel('z')
title({['Pu = ',num2str(Puvw(1)),...
       ', Pv = ',num2str(Puvw(2)),...
       ', Pw = ',num2str(Puvw(3))];...
       ['Px = ',num2str(Pxyz(1)),...
       ', Py = ',num2str(Pxyz(2)),...
       ', Pz = ',num2str(Pxyz(3))]})
%% s.77 kamera   
clear
%yksikkökuution etuosa
P=[1,0,0
   1,1,0
   1,1,1
   1,0,1
   1,0,0
   1,0,1
   0,0,1
   0,1,1
   1,1,1
   1,1,0
   0,1,0
   0,1,1]'
Ne=length(P(1,:))
P=[P;ones(1,Ne)];
%takaosa
Q=[1,0,0
   0,0,0
   0,0,1
   0,0,0
   0,1,0]'
Nt=length(Q(1,:))
Q=[Q;ones(1,Nt)];
eye=[5,4,2]'
look=[0.5,0.5,0.5]'
up=[0,0,1]'
N=3


w=eye-look
u=cross(up,w)
v=cross(w,u)
u=u/norm(u)
v=v/norm(v)
w=w/norm(w)
P0=eye
K=[u,v,w,P0;0,0,0,1]
%etuosan uvw-koordinaatit
Puvw=K^-1*P;
Pu=Puvw(1,:);
Pv=Puvw(2,:);
Pw=Puvw(3,:);
%koordinaatit kuvatasolla
KPu=N*Pu./(-Pw);
KPv=N*Pv./(-Pw);

%takaosan uvw-koordinaatit
Quvw=K^-1*Q;
Qu=Quvw(1,:);
Qv=Quvw(2,:);
Qw=Quvw(3,:);
%koordinaatit kuvatasolla
KQu=N*Qu./(-Qw);
KQv=N*Qv./(-Qw);

L=1 %akseleiden rajat

figure(1)
plot(KQu,KQv,'r.-','linewidth',1.0,'markersize',15)
hold
plot(KPu,KPv,'b.-','linewidth',2.5,'markersize',20)
hold off
axis([-L,L,-L,L])
axis square
grid

%kuvatason pisteet
O=eye-N*w;
llc=O-L*u-L*v %vasen ala
luc=O-L*u+L*v %vasen ylä
rlc=O+L*u-L*v %oikea ala
ruc=O+L*u+L*v %oikea ylä
pp=[llc,luc,ruc,rlc,llc]; 

figure(2)
plot3(Q(1,:),Q(2,:),Q(3,:),'r.-','linewidth',1.5,'markersize',15)
hold
plot3(P(1,:),P(2,:),P(3,:),'b.-','linewidth',2.5,'markersize',20)
plot3([0,2],[0,0],[0,0],'k')
plot3([0,0],[0,2],[0,0],'k')
plot3([0,0],[0,0],[0,2],'k')
p1=plot3(eye(1),eye(2),eye(3),'k.','markersize',20)
p2=plot3(look(1),look(2),look(3),'m.','markersize',20)
plot3([eye(1),look(1)],[eye(2),look(2)],[eye(3),look(3)])
p3=plot3([O(1),O(1)+u(1)],[O(2),O(2)+u(2)],[O(3),O(3)+u(3)],'m','linewidth',2)
p4=plot3([O(1),O(1)+v(1)],[O(2),O(2)+v(2)],[O(3),O(3)+v(3)],'g','linewidth',2)
plot3(O(1)+KPu*u(1)+KPv*v(1),...
      O(2)+KPu*u(2)+KPv*v(2),...
      O(3)+KPu*u(3)+KPv*v(3),'b','linewidth',2)
plot3(O(1)+KQu*u(1)+KQv*v(1),...
      O(2)+KQu*u(2)+KQv*v(2),...
      O(3)+KQu*u(3)+KQv*v(3),'r','linewidth',1)   
k=1
plot3([eye(1),P(k,1)],[eye(2),P(k,2)],[eye(3),P(k,3)])
p5=plot3(O(1),O(2),O(3),'.','markersize',20)
plot3(pp(1,:),pp(2,:),pp(3,:),'linewidth',1.5)
hold off
grid
axis equal
view(eye)
legend([p1,p2,p3,p4,p5],{'eye','look','u','v','O'},'fontsize',12)
xlabel('x')
ylabel('y')
   