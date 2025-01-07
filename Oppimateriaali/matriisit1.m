%% s.1-2
clear

A=[4,2
   1,5] 

[m,n]=size(A) %A:n koko, mxn

A(1,2) %alkio vaakarivillä 1, pystyrivillä 2


A(1,:) %vaakarivi 1
A(:,2) %pystyrivi 2

%% vaaka- ja pystyvektoreille riittää yksi indeksi
v=[3,2,4,-5] %vaakavektori, 1x4
v=[3;2;4;-5] %pystyvektori, 4x1
v(1) %alkio 1
n=length(v) %v:n pituus

%% matriisi voidaan muodostaa myös isommista palasista

A=[4,2
   1,5] 

B=[7,3,-6
   0,9,8]

C=[2,5;0,-4;1,7]


D=[A,B] %samankorkuiset A ja B vierekkäin
E=[A;C] %samanlevyiset A ja C päällekkäin
%% riveittäin tai sarakkeittain
F=zeros(2,3); %2x3-matriisi nollia
F(1,:)=[1,4,3] %rivi 1
F(2,:)=[9,5,2] %rivi 2

G=zeros(3,3);
G(:,1)=[3;2;8] %sarake 1
G(:,2)=[0;5;6] %sarake 2
G(:,3)=[2;2;1] %sarake 3 
%% s.4 datamatriisi
%tiedosto data.txt oletushakemistossa (current folder)
%luetaan matriisiksi X
clear
X=importdata('data.txt');
[m,n]=size(X)
r=1
X(r,:) %rivi r
%%
s=3 %sarake s
plot(1:m,X(:,s),'b.-','markersize',12)
grid
title(['sarake ',num2str(s)])
xlabel('datapiste')

%% s.5-6, kuva.png oletushakemistossa (current folder)
clear 
close all
%luetaan kuva matriisiksi M
M=imread('kuva.png');

[m,n]=size(M)
%M:n alkiot 8-bittisiä kokonaislukuja (uint8) 0,1,...,255
%0 = musta, 255 = valkea

M(1,1:10) %rivi 1, sarakkeet 1,2,...,10

%% kuva
figure(1)
imshow(M) %0 = musta, 255 = valkea
axis on %näyttää arvot akseleilla

%%
M(1:50,1:50)=125; %vasen ylänurkka valkoiseksi
M(230:320,410:512)=0; %valkosipuli mustaksi
M(47:54,:)=127
M(:,134:172)=127
%% kuva
figure(2)
imshow(M) %0 = musta, 1 = valkea
axis on %näyttää arvot akseleilla

%%
r=1
s=300
Mr=M(r,:); %rivi r
Ms=M(:,s); %sarake s

figure(3)
subplot(2,1,1)
plot(1:n,Mr,'r')
grid
title(['rivi r = ',num2str(r)])
xlim([1,n])

subplot(2,1,2)
plot(1:m,Ms,'b')
grid
title(['sarake s = ',num2str(s)])
xlim([1,m])

%% s.9 transpoosi
clear
A=[4,2
   2,0
  -2,7]

A' %A^T

[1,2,3]'  %vaaka -> pysty
[1;2;3]'  %pysty -> vaaka

%% s.11-12 , alkioittaiset laskutoimitukset
clear
A=[4,2;1,5]
B=[2,5;0,-4]

A+B %alkioittainen yhteenlasku
A-B %alkiottainen vähennyslasku
3*A %kerrotaan jokainen A:n alkio 3:lla 
A/3 %jaetaan jokainen A:n alkio 3:lla
A+3 %lisätään jokaiseen A:n alkioon 3
A-3 %vähennetään jokaisesta A:n alkiosta 3


%alkioittaiset kerto- ja jakolasku ja potenssiin korotus !!!!!!!!!!!!
A.*B %.* kerrotaan samankokoiset A:n ja B:n alkioittain
A./B %./ jaetaan samankokoiset A:n ja B:n alkioittain
3./A %./ jaetaan 3 jokaisella A:n alkiolla, alkio i,j = 3/A(i,j) 
A.^2 %.^ korotetaan A alkioittain potenssiin
%% s.13, etsitään kuvasta reunoja
% reuna = vierekkäisten pikseleiden arvojen ero on 
% suuri (plus- tai miinusmerkkinen)
clear
close all
M=imread('kuva.png');
[m,n]=size(M)

M(1,1:10)

%muutetaan arvot liukuluvuiksi (double) 
%ja skaalataan arvot välille 0 ... 1
M=double(M)/255;

M(1,1:10)

%%

r=150 %rivi
s=300 %sarake

figure(1)
imshow(M)
hold on
plot([0,n],[r,r],'r')
plot([s,s],[0,m],'b')
hold off
axis on
%%
Mr=M(r,:); %rivi r
Rr=Mr(1:n-1)-Mr(2:n); %rivi r, vasen-oikea
%Mr(1:n-1)=[Mr(1),Mr(2),...,Mr(n-1)]
%  Mr(2:n)=[Mr(2),Mr(3),...,Mr(n)]

Ms=M(:,s); %sarake s
Rs=Ms(1:m-1)-Ms(2:m);%sarake s, ylempi-alempi

figure(2)
subplot(2,1,1)
plot(1:n,Mr,'r')
grid
title(['rivi r = ',num2str(r)])
xlim([1,n])

subplot(2,1,2)
plot(1:n-1,Rr,'r')
grid
title('vierekkäisten pikseleiden erotukset M(r,s)-M(r,s+1)')
xlim([1,n])

figure(3)
subplot(2,1,1)
plot(1:m,Ms,'b')
grid
title(['sarake s = ',num2str(s)])
xlim([1,m])

subplot(2,1,2)
plot(1:m-1,Rs,'b')
xlim([1,m])
grid
title('päällekkäisten pikseleiden erotukset M(r,s)-M(r+1,s)')

%%
%vaakasuorat reunat eli päällekkäisten pikseleiden 
%erotukset, ylempi-alempi, (m-1) x n - matriisi
Mvr=M(1:m-1,:)-M(2:m,:);

%M(1:m-1,:) = M:n rivit 1...m-1
%M(2:m,:) = M:n rivit 2...m 


%pystysuorat reunat eli vierekkäisten pikseliden
%erotukset, vasen-oikea, m x (n-1) - matriisi
Mpr=M(:,1:n-1)-M(:,2:n);

%itseisarvot
Mvr=abs(Mvr); 
Mpr=abs(Mpr);

figure(4)
imshow(-Mvr,[]) %[]: min = musta, max = valkea 
axis on
title('vaakasuorat reunat, -|Mvr|')

figure(5)
imshow(-Mpr,[]) 
axis on
title('pystysuorat reunat, -|Mpr|')

%%
%vaaka- ja pystysuorat reunat
Mvpr=Mvr(:,1:n-1)+Mpr(1:m-1,:);


figure(6)
imshow(-Mvpr,[]) 
axis on
title('vaaka- ja pystysuorat reunat, -Mvpr')


%% s.19, broadcasting
clear
A=[1,2
   3,4
   5,6]
u=[7,8]
v=[9;10;11]

A+u
A+v
u+v

%% s.21 datan skaalaus
%tiedosto data.txt oletushakemistossa (current folder)
%luetaan matriisiksi X
clear
%close all
X=importdata('data.txt');


m=min(X) %sarakkeiden minimit
M=max(X) %sarakkeiden maksimit
Xhat=(X-m)./(M-m); %välille 0...1 skaalatut datapisteet

min(Xhat)
max(Xhat)

X(1:5,:)
Xhat(1:5,:)

s=1 %sarake nro

figure(1)
subplot(2,1,1)
plot(X(:,s),'linewidth',1)
grid
title(['alkuperäinen sarake ',num2str(s)])

subplot(2,1,2)
plot(Xhat(:,s),'linewidth',1)
grid
title(['skaalattu sarake ',num2str(s),',  min = 0, max = 1 '])

%% skaalaus 2
mu=mean(X) %sarakkeiden keskiarvot
sigma=std(X,1) %sarakkeiden keskihajonnat

Xhat2=(X-mu)./sigma; %sarakkeiden ka=0, kh=1

mean(Xhat2)
std(Xhat2,1)

X(1:5,:)
Xhat(1:5,:)


s=1 %sarake nro

figure(2)
subplot(2,1,1)
plot(X(:,s),'linewidth',1)
grid
title(['alkuperäinen sarake ',num2str(s)])

subplot(2,1,2)
plot(Xhat2(:,s),'linewidth',1)
grid
title(['skaalattu sarake ',num2str(s),', ka = 0, kh = 1'])




%% s.31, matriisien kertolasku
clear
close all
A=[4,2
   1,5]
B=[7,3,-6
   0,9,8]
C=[2,5
   0,-4
   1,7]

AB=A*B
CA=C*A
BC=B*C
CB=C*B

%B*A
%A*C

A^2 %A*A

%% s.39 sekoitus
clear
close all
%sekoitusmatriisi
M=[0.85,0.30,0.15
   0.1,0.45,0.20
   0.05,0.25,0.65]

%alkutilanne
X0=[100;100;100]
%kierrosten määrä
N=50
%matriisi, johon määrät kerätään
X=zeros(3,N+1);
X(:,1)=X0;

Xkm1=X0; %X_{k-1}, edelliset määrät
for k=1:N
Xk=M*Xkm1; %uudet määrät 
X(:,k+1)=Xk;
Xkm1=Xk;
end

Xk/sum(X0) %osuudet
%%
plot(0:N,X(1,:),'r.-','linewidth',1.5,'markersize',15)
hold
plot(0:N,X(2,:),'g.-','linewidth',1.5,'markersize',15)
plot(0:N,X(3,:),'b.-','linewidth',1.5,'markersize',15)
hold off
grid
xlabel('kierros k')
legend({'A_k','B_k','C_k'},'fontsize',10)
title(['Alkutilanne: ',num2str(X0')])
%%
k=30
M^k  

%% s.43 PageRank
clear 
n=4 %sivujen lukumäärä
d=0.85 %damping factor

%siirtymätodennäköisyydet
%P_ij = siirtymä j->i
%vaihe 1
P1=[  0,   0,  1, 1/2
    1/3,   0,  0,   0
    1/3, 1/2,  0, 1/2
    1/3, 1/2,  0,   0]

%vaihe 2
P2=ones(n,n)/n %P2_ij=1/n

%PageRank-matriisi
P=d*P1+(1-d)*P2

%aloitussivu
X0=ones(n,1)/n %[1/n;1/n;...;1/n]

N=20 %kierrosten määrä
%matriisi, johonsiirtymätodennäköisyydet kerätään
X=zeros(n,N+1);
X(:,1)=X0;


Xkm1=X0; %edelliset tn:t 
for k=1:N
    Xk=P*Xkm1; %uudet tn:t
    X(:,k+1)=Xk;
    Xkm1=Xk;    
end

%%
plot(0:N,X(1,:),'r.-','linewidth',1.5,'markersize',15)
hold
plot(0:N,X(2,:),'g.-','linewidth',1.5,'markersize',15)
plot(0:N,X(3,:),'b.-','linewidth',1.5,'markersize',15)
plot(0:N,X(4,:),'m.-','linewidth',1.5,'markersize',15)
hold off
grid
xlabel('askel k')
legend({'1','2','3','4'},'fontsize',10)
%%
k=20
P^k
%% s.48 graafin linkkimatriisi 2
clear
A=[0,1,0,1,0
   1,0,0,0,0
   0,1,1,0,0
   0,0,1,0,1
   1,0,1,0,0]

A^2
A^3
A^10
%% s.50 käänteismatriisi
clear
A=[1,2
   3,4]
A^-1
inv(A)

B=[1,2
   2,4]
B^-1
inv(B)
%% s.61 yhtälöparin geometria 1
clear
close all
%kerroinmatriisi
A=[1,1
  -1,2]

%oikea puoli
B=[3
   2]

X=A^-1*B %tai A\B
x=X(1)
y=X(2)
%%
%A:n rivivektorit
u=A(1,:)
v=A(2,:)

%suoran yhtälö 
%ax+by=c
%y=1/b*(c-ax)
%normaali n=[a,b]

L=4 %suoran pituus = 2L
xs=[x-L/2,x+L/2]

%s1, normaali u
a=u(1)
b=u(2)
c=B(1)
ys1=1/b*(c-a*xs);
%s2, normaali v
a=v(1)
b=v(2)
c=B(2)
ys2=1/b*(c-a*xs);

plot(xs,ys1,'r','linewidth',2)
hold
plot(xs,ys2,'b','linewidth',2)
plot(x,y,'k.','markersize',20)
hold off
grid
legend('suora 1','suora 2','X')
%% s.62 yhtälöparin geometria 2
clear
close all
%kerroinmatriisi
A=[1,1
  -1,2]
%oikea puoli
B=[3
   2]
%ratkaisu
X=A^-1*B
x=X(1)
y=X(2)

%A:n sarakevektorit
u=A(:,1)
v=A(:,2)

P=x*u
Q=y*v

figure(1)
plot([0,u(1)],[0,u(2)],'r','linewidth',3)
hold
plot([0,v(1)],[0,v(2)],'b','linewidth',3)
plot(B(1),B(2),'k.','markersize',20)
plot([0,P(1)],[0,P(2)],'r','linewidth',1)
plot([0,Q(1)],[0,Q(2)],'b','linewidth',1)
plot([P(1),B(1)],[P(2),B(2)],'b','linewidth',1)
plot([Q(1),B(1)],[Q(2),B(2)],'r','linewidth',1)
hold off
grid
axis equal
legend({'u','v','B'},'fontsize',12)
title({['x = ',num2str(x,3),', y = ',num2str(y,3)]},'fontsize',12)
%% s.63 yhtälöryhmän geometria 1
clear
close all
%kerroinmatriisi
A=[2,-1,3
   6,4,-2
  -2,2,1]
%oikea puoli
B=[1
   2
  -3]
%ratkaisu
X=A^-1*B
x=X(1)
y=X(2)
z=X(3)


%A:n rivivektorit
u=A(1,:)
v=A(2,:)
w=A(3,:)

%tason yhtälö 
%ax+by+cz=d
%z=1/c*(d-ax-by)
%normaali n=[a,b,c]

L=4 %tasojen koko LxL
xt=[x-L/2,x+L/2]
yt=[y-L/2,y+L/2]
[xt,yt]=meshgrid(xt,yt)

%t1, normaali u
a=u(1)
b=u(2)
c=u(3)
d=B(1)
zt1=1/c*(d-a*xt-b*yt);
%t2, normaali v
a=v(1)
b=v(2)
c=v(3)
d=B(2)
zt2=1/c*(d-a*xt-b*yt);
%t3, normaali w
a=w(1)
b=w(2)
c=w(3)
d=B(3)
zt3=1/c*(d-a*xt-b*yt);

surf(xt,yt,zt1,'facecolor','r')%,'edgecolor','none')
hold
surf(xt,yt,zt2,'facecolor','g','edgecolor','none')
surf(xt,yt,zt3,'facecolor','b','edgecolor','none')
plot3(x,y,z,'k.','markersize',30)
hold off
grid on
alpha(0.6)
legend('taso1','taso2','taso3','X')
xlabel('x')
ylabel('y')

%% s.64 yhtälöryhmän geometria 2
A=[2,-1,3
   6,4,-2
  -2,2,1]
B=[1
   2
  -3]
X=A^-1*B
x=X(1)
y=X(2)
z=X(3)

%A:n sarakevektorit
u=A(:,1)
v=A(:,2)
w=A(:,3)

P=x*u
Q=P+y*v
R=Q+z*w

figure(2)
plot3([0,u(1)],[0,u(2)],[0,u(3)],'r','linewidth',3)
hold
plot3([0,v(1)],[0,v(2)],[0,v(3)],'g','linewidth',3)
plot3([0,w(1)],[0,w(2)],[0,w(3)],'b','linewidth',3)
plot3(B(1),B(2),B(3),'k.','markersize',20)
plot3([0,P(1)],[0,P(2)],[0,P(3)],'r','linewidth',1)
plot3([P(1),Q(1)],[P(2),Q(2)],[P(3),Q(3)],'g','linewidth',1)
plot3([Q(1),R(1)],[Q(2),R(2)],[Q(3),R(3)],'b','linewidth',1)
L=5
plot3([-L,L],[0,0],[0,0],'k')
plot3([0,0],[-L,L],[0,0],'k')
plot3([0,0],[0,0],[-L,L],'k')
hold off
grid
xlabel('x')
ylabel('y')
zlabel('z')
legend({'u','v','w','B'},'fontsize',12)
title({['x = ',num2str(x,3),', y = ',num2str(y,3),', z = ',num2str(z,3)]},'fontsize',12)
%% s.65 virtapiiri
clear
R1=1
R2=2
R3=3
R4=4
R5=5
R6=6
V=7

A=[R1+R2+R3,-R2,0
   R2,-(R2+R4),R4
   0,R4,-(R4+R5+R6)]
B=[V;0;0]
X=A\B

%% s.67, pisteiden kautta kulkeva paraabeli 
clear
close all
%pisteet
P=[1,3
   3,5
   4,4]
x=P(:,1)
y=P(:,2)
n=length(x)
A=[x.^2,x,ones(n,1)] 
X=A^-1*y %tai A\y
a=X(1)
b=X(2)
c=X(3)
%% kuvaaja välillä x=xv...xo
xv=x(1)-1 %v=vasen
xo=x(n)+1 %o=oikea 
xx=xv:(xo-xv)/100:xo;
yy=a*xx.^2+b*xx+c; 

figure(1)
plot(xx,yy,'b','linewidth',2)
hold on
plot(x,y,'r.','markersize',20)
hold off
grid
xlabel('x')


%% s.69, 3 pisteen kautta kulkeva ympyrä
clear
close all
P1=[1,1]
P2=[2,4]
P3=[5,2]

A=2*[P3-P1
     P3-P2]
B=[norm(P3)^2-norm(P1)^2
   norm(P3)^2-norm(P2)^2]
X=A^-1*B %tai A\B
%%
r=norm(P1-X')

t=0:360;
x=X(1)+r*cosd(t);
y=X(2)+r*sind(t);

%suora s13
L=2 %suoran pituus = 2*L
P1P3=P3-P1
n1=P1P3/norm(P1P3)
s1=[-n1(2),n1(1)]
P13=(P1+P3)/2
s13v=P13+L*s1
s13o=P13-L*s1
%s23
P2P3=P3-P2
n2=P2P3/norm(P2P3)
s2=[-n2(2),n2(1)]
P23=(P2+P3)/2
s23v=P23+L*s2
s23o=P23-L*s2
%s12
P1P2=P2-P1
n3=P1P2/norm(P1P2)
s3=[-n3(2),n3(1)]
P12=(P1+P2)/2
s12v=P12+L*s3
s12o=P12-L*s3

P=[P1;P2;P3;P1]

plot(x,y,'r','linewidth',2)
hold
plot(P(:,1),P(:,2),'k.','markersize',20)
plot(P(:,1),P(:,2))
p1=plot([s13v(1),s13o(1)],[s13v(2),s13o(2)],'linewidth',1)
p2=plot([s23v(1),s23o(1)],[s23v(2),s23o(2)],'linewidth',1)
p3=plot([s12v(1),s12o(1)],[s12v(2),s12o(2)],'linewidth',1)
p4=plot(X(1),X(2),'r.','markersize',20)
hold off
grid
axis equal
legend([p1,p2,p3,p4],{'s_{13}','s_{23}','s_{12}','X'},'fontsize',12)

%% s.71, 2D-paikannus, 3 tukiasemaa, tapa 1
clear
close all
P1=[0,0]
P2=[6,0]
P3=[3,5]
P=[1,4]

%mitatut etäisyydet
r1=norm(P1-P)+0.3
r2=norm(P2-P)-0.8
r3=norm(P3-P)+0.4

A=2*[P3-P1
     P3-P2]
B=[r1^2-r3^2-norm(P1)^2+norm(P3)^2
   r2^2-r3^2-norm(P2)^2+norm(P3)^2]
X=A^-1*B
%%
t=0:360;
c=cosd(t);
s=sind(t);
y1x=P1(1)+r1*c;
y1y=P1(2)+r1*s;
y2x=P2(1)+r2*c;
y2y=P2(2)+r2*s;
y3x=P3(1)+r3*c;
y3y=P3(2)+r3*s;

%suorat s12,s13,s23
%s13
L=5 %suorien pituus = 2*L
P1P3=P3-P1
p1p3=norm(P1P3)
L1=1/2*p1p3+1/2*(r1^2-r3^2)/p1p3
A13=P1+L1*P1P3/p1p3 %suoran s13 piste janalla P1P3
s13=[-P1P3(2),P1P3(1)]/p1p3 %suoran suuntainen yksikkövektori
P13v=A13-L*s13  %s13:n päätepiste
P13o=A13+L*s13  %s13:n päätepiste
%s23
P2P3=P3-P2
p2p3=norm(P2P3)
L1=1/2*p2p3+1/2*(r2^2-r3^2)/p2p3
A23=P2+L1*P2P3/p2p3
s23=[-P2P3(2),P2P3(1)]/p2p3
P23v=A23-L*s23
P23o=A23+L*s23
%s12
P1P2=P2-P1
p1p2=norm(P1P2)
L1=1/2*p1p2+1/2*(r1^2-r2^2)/p1p2
A12=P1+L1*P1P2/p1p2
s12=[-P1P2(2),P1P2(1)]/p1p2
P12v=A12-L*s12
P12o=A12+L*s12

plot(y1x,y1y,'r','linewidth',1.5)
hold
plot(y2x,y2y,'g','linewidth',1.5)
plot(y3x,y3y,'b','linewidth',1.5)
p1=plot(P1(1),P1(2),'r.','markersize',20)
p2=plot(P2(1),P2(2),'g.','markersize',20)
p3=plot(P3(1),P3(2),'b.','markersize',20)
p5=plot([P13v(1),P13o(1)],[P13v(2),P13o(2)],'m','linewidth',1.5)
p6=plot([P23v(1),P23o(1)],[P23v(2),P23o(2)],'c','linewidth',1.5)
p7=plot([P12v(1),P12o(1)],[P12v(2),P12o(2)],'linewidth',1.0)
p4=plot(X(1),X(2),'k.','markersize',20)
hold off
grid
axis([-5,12,-7,10])
axis square
set(gca,'xtick',-4:2:10)
legend([p1,p2,p3,p4,p5,p6,p7],...
       {'P_1','P_2','P_3','X','s_{13}','s_{23}','s_{12}'},...
        'fontsize',12)
title({['r_1 = ',num2str(r1,3),', r_2 = ',num2str(r2,3),...
    ', r_3 = ',num2str(r3,3),...
       ', X = [',num2str(X(1),3),',',num2str(X(2),3),']^T']})



%% s.79-, 3x3-lämpötila
clear
close all 

Ty=[100,100,100] %yläreunan lämpötilat
Tv=[0,0,0] %vasen reuna
To=Tv %oikea reuna
Ta=Tv %alareuna

K=[4 -1 0
   -1 4 -1
   0 -1 4]
I=eye(3) %3x3 yksikkömatriisi

O=zeros(3,3) %3x3-matriisi nollia

A=[K,-I,O
  -I,K,-I
   O,-I,K]

B=[Tv(1)+Ty(1);Ty(2);Ty(3)+To(1);Tv(2);0;To(2);Tv(3)+Ta(1);Ta(2);Ta(3)+To(3)]

T=A^-1*B %tai A\B
%%
%kuva, neliön nurkat pisteissä [0,0],[1,0],[1,-1],[0,-1]
N=3
d=1/(N+1) %hilapisteiden väli
xh=d:d:1-d; %hilapisteiden x:t
yh=-d:-d:-1+d; %ja y:t



plot([0 1 1 0 0],[0 0 -1 -1 0],'k') %neliö
hold on
%mustat pallerot reunoille
plot(xh,zeros(1,N),'k.','markersize',15)
plot(xh,-ones(1,N),'k.','markersize',15)
plot(zeros(1,N),yh,'k.','markersize',15)
plot(ones(1,N),yh,'k.','markersize',15)

%reunojen lämpötilat
for k=1:N
   text(xh(k)-0.03,0.05,num2str(Ty(k))) 
   text(xh(k)-0.01,-1-0.05,num2str(Ta(k)))
   text(-0.05,yh(k),num2str(Tv(k)))
   text(1.03,yh(k),num2str(To(k)))
end

%hilapisteet
n=1 
for k=1:N
   for m=1:N
       plot(xh(m),yh(k),'r.','markersize',15)
       text(xh(m)-0.03,yh(k)-0.02,num2str(T(n),4))
       n=n+1;
   end
end
hold off
grid
axis([-0.1 1.1 -1.1 0.1])
axis square
set(gca,'xticklabel',{})
set(gca,'xtick',xh)
set(gca,'ytick',-1+d:d:-d)
set(gca,'yticklabel',{})

%% s.85, jousi-massa-systeemi
clear
close all

%reduced incidence matrix
A=[1,0,0
   -1,1,0
   0,-1,1
   0,0,-1]
%jousivakiot
c1=1
c2=2
c3=3
c4=4

C=diag([c1,c2,c3,c4]) %diagonaali- eli lävistäjämatriisi

%voimat
f1=1
f2=2
f3=3
f=[f1;f2;f3]

%jäykkyysmatriisi
K=A'*C*A
%Ku=f -> u=K^-1*f
u=inv(K)*f %massojen liikkeet (tai K^-1*f tai K\f)
v=A*u %jousien venymät
y=C*v %jousivoimat
%% kuva
L=2 %jousien alkupituus
subplot(1,2,1)
plot([0,0],[0,-L],'r','linewidth',2)
hold
plot([0,0],[-L,-2*L],'g','linewidth',2)
plot([0,0],[-2*L,-3*L],'b','linewidth',2)
plot([0,0],[-3*L,-4*L],'m','linewidth',2)
plot([0,0,0],[-L,-2*L,-3*L],'k.','markersize',30)
hold off
grid
xlim([-1,1])


subplot(1,2,2)
plot([0,0],[0,-L-u(1)],'r','linewidth',2)
hold
plot([0,0],[-L-u(1),-2*L-u(2)],'g','linewidth',2)
plot([0,0],[-2*L-u(2),-3*L-u(3)],'b','linewidth',2)
plot([0,0],[-3*L-u(3),-4*L],'m','linewidth',2)
plot([0,0,0],[-L-u(1),-2*L-u(2),-3*L-u(3)],'k.','markersize',30)
hold off
grid
xlim([-1,1])


%% s.93- , virtapiiri
clear
close all

%incidence matrix
A=[1,-1,0,0
   1,0,-1,0
   1,0,0,-1
   0,1,0,-1
   0,0,1,-1]

g=4 %ground node, ug=0
%reduced incidence matrix
A(:,g)=[] %poistetaan A:sta sarake g -> reduced incidence matrix

%resistanssit
R1=0.5
R2=0.5
R3=0.25
R4=0.25
R5=0.25
R=[R1,R2,R3,R4,R5]
c=1./R
C=diag(c)

%virrat
f1=1
f2=0
f3=0
f=[f1;f2;f3]

K=A'*C*A %resistivity matrix

u=K^-1*f %potentiaalit
v=A*u %jännitteet
y=C*v %virrat 

%% s.95, virtapiiri
clear
%incidence matrix
A=[-1,0,0,1
   1,0,-1,0
   0,1,-1,0
   0,0,1,-1
   1,-1,0,0
   0,-1,0,1]
g=4 %ground node,  potentiaali = 0
%reduced incidece matrix
A(:,g)=[] %poistetaan sarake g
R=[1,30,1,55,25,50] %resistanssit
c=1./R
C=diag(c)
b=[10,0,0,0,0,0]' %batteryt
K=A'*C*A %resistivity matrix
u=inv(K)*(-A'*C*b)%potentiaalit
v=A*u+b %jännitteet
y=C*v   %virrat


%% s.103-  PNS-ratkaisu
clear
close
A=[2,3
   1,-4 
   3,1]

B=[5;1;2]

%PNS-ratkaisu 
X=A\B
A*X-B %virheet
norm(A*X-B)^2 %virheiden neliöiden summa
%% s.105, geometria 1
%piirretään suorat ja PNS-ratkaisun X
%kohtisuorat projektiot suorille

L=5 %suorien pituus = 2L
col=['r','g','b'] %suorien värit
figure(2)
for k=1:3
%suora k    
a=A(k,1);
b=A(k,2);
c=B(k);
n=[a;b]; %suoran normaali
n0=n/norm(n);
P=c/norm(n)*n0; %piste suoralla
nk=[-n(2);n(1)]; %n:ää vastaan kohtisuora vektori
nk0=nk/norm(nk);
%suoran ax+bx=c päätepisteet
P1=P+L*nk0
P2=P-L*nk0
%lasketaan Q = X:n projektio suoralle k 
PX=X-P;
QX=dot(PX,n0)*n0; %n:n suuntainen komponentti
Q=X-QX;
d(k)=norm(QX)*norm(n); %yhtälön k virhe
p(k)=plot([P1(1),P2(1)],[P1(2),P2(2)],col(k),'linewidth',2);
if k==1
    hold on
end
q(k)=plot([Q(1),X(1)],[Q(2),X(2)],'linewidth',2);
end
r=plot(X(1),X(2),'k.','markersize',20);

hold off
grid
axis([-4,4,-4,4])
axis square
legend([p,q,r],{'s_1','s_2','s_3','d_1','d_2','d_3','X'},...
       'fontsize',12)

%yhtälöiden virheet
A*X-B
d'

%virheiden neliösumma
sum(d.^2)
%% s.107 geometria 2
u=A(:,1); %sarake 1
v=A(:,2); %sarake 2

%taso uv
s=[-2,3];
t=s;
[s,t]=meshgrid(s,t);
xt=s*u(1)+t*v(1);
yt=s*u(2)+t*v(2);
zt=s*u(3)+t*v(3);

AX=A*X;
x=X(1)
y=X(2)
surf(xt,yt,zt,'facecolor','c')
hold on
p1=plot3([0,u(1)],[0,u(2)],[0,u(3)],'r','linewidth',3)
p2=plot3([0,v(1)],[0,v(2)],[0,v(3)],'g','linewidth',3)
plot3([0,x*u(1)],[0,x*u(2)],[0,x*u(3)],'r','linewidth',1)
plot3([0,y*v(1)],[0,y*v(2)],[0,y*v(3)],'g','linewidth',1)
plot3([x*u(1),AX(1)],[x*u(2),AX(2)],[x*u(3),AX(3)],'g','linewidth',1)
plot3([y*v(1),AX(1)],[y*v(2),AX(2)],[y*v(3),AX(3)],'r','linewidth',1)
p3=plot3(B(1),B(2),B(3),'m.','markersize',20)
p4=plot3(AX(1),AX(2),AX(3),'k.','markersize',20)
plot3([AX(1),B(1)],[AX(2),B(2)],[AX(3),B(3)],'linewidth',2)
alpha(0.2) %läpinäkyvyys, ei toimi octavessa
hold off
grid on
axis equal
legend([p1,p2,p3,p4],'u','v','B','AX')
xlabel('x')
ylabel('y')
zlabel('z')
title(['AX = ',num2str(X(1)),' u + ',num2str(X(2)),' v'])

%% s.115, PNS-suora 
clear
close
%pisteet
P=[0 1
   1 4
   3 2
   5 5]
x=P(:,1)
y=P(:,2)
N=length(x)
A=[x,ones(N,1)]
B=y
X=A\B %(A'*A)^-1*A'*B

a=X(1)
b=X(2)

%suoran päätepisteet
xv=min(x)-2
xo=max(x)+2

yv=a*xv+b
yo=a*xo+b



figure(1)
plot(x,y,'b.','markersize',20)
hold on
plot([xv,xo],[yv,yo],'r','linewidth',2)
hold off
grid
title(['y=ax+b, a = ',num2str(a),', b = ',num2str(b)])
%% s.116, PNS-paraabeli

A=[x.^2,x,ones(N,1)]
B=y
X=A\B

a=X(1)
b=X(2)
c=X(3)

xv=min(x)-2
xo=max(x)+2
xx=xv:(xo-xv)/100:xo;
yy=a*xx.^2+b*xx+c;


figure(2)
plot(x,y,'b.','markersize',20)
hold on
plot(xx,yy,'r','linewidth',2)
hold off
grid
title(['y=ax^2+bx+c, a = ',num2str(a),...
      ', b = ',num2str(b),...
       ', c = ',num2str(c)])


%% s.117 , U=E-RI 
clear
close
%mittaustulokset I,U
P=[0.5,8.2
   1.1,7.1
   2.4,6.3
   3.9,5.4
   5.1,4.9]
x=P(:,1)
y=P(:,2)
N=length(x)
A=[x,ones(N,1)]
B=y
X=A\B %(A'*A)^-1*A'*B
%U=a*I+b=E-R*I
a=X(1)
b=X(2)
E=b
R=-a
%suoran päätepisteet
xv=min(x)-0.5
xo=max(x)+0.5

yv=a*xv+b
yo=a*xo+b

figure(1)
plot(x,y,'b.','markersize',20)
hold on
plot([xv,xo],[yv,yo],'r','linewidth',2)
hold off
grid
title(['U = E - R I, E = ',num2str(E),', R = ',num2str(R)])
xlabel('I')
ylabel('U','rotation',0)
%% s.123- PNS-laskukaava, data2.txt oletushakemistossa
clear
close all
data=importdata('data2.txt');
x=data(:,1);
y=data(:,2);
z=data(:,3);
n=length(x)

data(1:5,:) %5 ensimmäistä riviä
%%
%PNS-laskukaava z=ax^2+bxy+cy^2+dx+ey+f
A=[x.^2,x.*y,y.^2,x,y,ones(n,1)];
r=A\z
a=r(1)
b=r(2)
c=r(3)
d=r(4)
e=r(5)
f=r(6)
%%
%testipiste
x0=1.25
y0=0.4
z0=a*x0^2+b*x0*y0+c*y0^2+d*x0+e*y0+f
%%
figure(1)
plot(y,z,'r.','markersize',20)
hold
%paraabelit x=1.48,x=1.4,...,x=0.9
Y=0:0.01:1.2;
xx=[1.48,1.4,1.3,1.2,1.1,1.0,0.9];
nx=length(xx);
for k=1:nx
X=xx(k);   
Z=a*X^2+b*X*Y+c*Y.^2+d*X+e*Y+f;
p(k)=plot(Y,Z,'linewidth',1);
end
p(nx+1)=plot(y0,z0,'k.','markersize',20);

hold off
grid
legend(p,'1.48','1.4','1.3','1.2','1.1','1.0','0.9','y0,z0')
title(['x_0 = ',num2str(x0),', y_0 = ',...
        num2str(y0),', z_0 = ',num2str(z0)])
xlabel('y')
ylabel('z','rotation',0)
%% pinta z=ax^2+bxy+cy^2+dx+ey+f 
xx=0.8:0.1:1.6;
yy=-0.2:0.1:1.4;
[xx,yy]=meshgrid(xx,yy);
zz=a*xx.^2+b*xx.*yy+c*yy.^2+d*xx+e*yy+f;

figure(2)
surf(xx,yy,zz)
hold
plot3(x,y,z,'r.','markersize',20)
hold off
alpha(0.7)
grid on
xlabel('x')
ylabel('y')
zlabel('z')
title('z=ax^2+bxy+cy^2+dx+ey+f')
%%
zKaava=A*r;
[z,zKaava]


%% s.127, 2D-paikannus, 3 tukiasemaa, tapa 2
clear
close all
%tukiasemat
P1=[0,0]
P2=[6,0]
P3=[3,5]

P=[1,4]

%mitatut etäisyydet pisteeseen P
r1=norm(P1-P)+0.3
r2=norm(P2-P)-0.8
r3=norm(P3-P)+0.4


%alkuarvaus
X=[-4,-4]
%korjaus (jotta päästeen while-looppiin)
dX=[1,1]
%kierrosten max-määrä
N=10
%kerätään arviot talteen
rata=zeros(N+1,2);
rata(1,:)=X;
n=1 %kierrosnumero

while norm(dX)>0.01 & n<=N
   XP1=P1-X;
   XP2=P2-X;
   XP3=P3-X;
   %lasketut etäisyydet
   R1=norm(XP1);
   R2=norm(XP2);
   R3=norm(XP3);
   %yksikkövektorit
   a1=XP1/R1;
   a2=XP2/R2;
   a3=XP3/R3;
   %kerroinmatriisi
   A=[a1;a2;a3];
   %oikea puoli
   B=[R1-r1
      R2-r2
      R3-r3];
   %korjaus
   dX=A\B;
   %uusi arvio
   X=X+dX';
   rata(n+1,:)=X;
   n=n+1;
end
rata=rata(1:n,:);
%ympyrät
t=0:360;
c=cosd(t);
s=sind(t);
y1x=P1(1)+r1*c;
y1y=P1(2)+r1*s;
y2x=P2(1)+r2*c;
y2y=P2(2)+r2*s;
y3x=P3(1)+r3*c;
y3y=P3(2)+r3*s;



plot(P1(1),P1(2),'r.','markersize',20)
hold
plot(P2(1),P2(2),'g.','markersize',20)
plot(P3(1),P3(2),'b.','markersize',20)
plot(X(1),X(2),'k.','markersize',20)
plot(y1x,y1y,'r','linewidth',1.5)
plot(y2x,y2y,'g','linewidth',1.5)
plot(y3x,y3y,'b','linewidth',1.5)
plot(rata(:,1),rata(:,2),'k.-','linewidth',1.5,'markersize',12)
hold off
grid
axis equal
legend({'P_1','P_2','P_3','X'},'fontsize',12)
title({['n = ',num2str(n),', ||\Delta X|| = ',num2str(norm(dX),3),...
        ', X  = [',num2str(X(1),3),',',num2str(X(2),3),']']})
