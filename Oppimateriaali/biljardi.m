clear
close all
L=5 %kentän leveys = 2*L
K=5 %kentän korkeus = 2*K
r1=0.7 %pallon 1 säde
r2=1 %pallon 2 säde
P1=[2,0] %pallon 1 alkupiste
P2=[-2,-1] %pallon 2 alkupiste
m1=r1^2 %pallon 1 massa
m2=r2^2 %pallon 2 massa
v1=[-6,5] %pallon 1 alkunopeus
v2=[3,-7]%pallon 2 alkunopeus

N=50 %törmäysten määrä
osumasivu1=0 %pallon 1 viimeisin törmäys-sivu
osumasivu2=0 %pallon 2 viimeisin törmäys-sivu

P1ratax=[] %keskipisteen 1 x-koordinaatit animaatioon
P1ratay=[] %keskipisteen 1 y-koordinaatit animaatioon
P2ratax=[] %keskipisteen 2 x-koordinaatit animaatioon
P2ratay=[] %keskipisteen 2 y-koordinaatit animaatioon

dt=0.01 %aika-askel, jolla keskipisteitä seurataan animaatiossa

for k=1:N
%P1
   %lasketaan osumahetket ja -pisteet kentän reunoihin 
   %suorien A,v ja C,w leikkauspiste A+tmin1*v
   A=P1;
   v=v1;
   tmin1=10^10; 
   %lasketaan kimpoamispiste
   %oikea sivu   
   C=[L-r1,-K];
   w=[0,1];
   wk=[-w(2),w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t1=dot(C-A,wk)/alakerta;
   else 
       t1=10^11;
   end
   if t1>=0 & t1<tmin1 & ~(osumasivu1 ==1);
       tmin1=t1;
       uusiosumasivu1=1;
       uusiP1=P1+t1*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv1=vw-vwk;
   end
    
   %yläsivu
   C=[L,K-r1];
   w=[-1,0];
   wk=[-w(2) w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t2=dot(C-A,wk)/alakerta;
   else 
       t2=10^11;
   end
   if t2>=0 & t2<tmin1 &  ~(osumasivu1 ==2);
       uusiosumasivu1=2;
       tmin1=t2;
       uusiP1=P1+t2*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv1=vw-vwk;
   end
  %vasen sivu
   C=[-L+r1,K];
   w=[0,-1];
   wk=[-w(2),w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t3=dot(C-A,wk)/alakerta;
   else 
       t3=10^11;
   end
   if t3>=0 & t3<tmin1 &  ~(osumasivu1 ==3)
       uusiosumasivu1=3;
       tmin1=t3;
       uusiP1=P1+t3*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv1=vw-vwk;
   end
   %alasivu
   C=[-L,-K+r1];
   w=[1 0];
   wk=[-w(2) w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t4=dot(C-A,wk)/alakerta;
   else 
       t4=10^11;
   end
   if t4>=0 & t4<tmin1 & ~(osumasivu1 ==4)
       uusiosumasivu1=4;
       tmin1=t4;
       uusiP1=P1+t4*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv1=vw-vwk;
   end
    


 %P2
 %lasketaan osumahetket ja -pisteet kentän reunoihin 
 %suorien A,v ja C,w leikkauspiste A+tmin2*v
  A=P2;
  v=v2;
  tmin2=10^10;
  
   %oikea sivu
   C=[L-r2,-K];
   w=[0,1];
   wk=[-w(2),w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t1=dot(C-A,wk)/alakerta;
   else 
       t1=10^11;
   end
   if t1>=0 & t1<tmin2 & ~(osumasivu2 ==1)
       tmin2=t1;
       uusiosumasivu2=1;
       uusiP2=P2+t1*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv2=vw-vwk;
   end
    
   %yläsivu
   C=[L,K-r2];
   w=[-1,0];
   wk=[-w(2),w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t2=dot(C-A,wk)/alakerta;
   else 
       t2=10^11;
   end
   if t2>=0 & t2<tmin2 &  ~(osumasivu2 ==2)
       uusiosumasivu2=2;
       tmin2=t2;
       uusiP2=P2+t2*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv2=vw-vwk;
   end
  %vasen sivu
   C=[-L+r2,K];
   w=[0,-1];
   wk=[-w(2),w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t3=dot(C-A,wk)/alakerta;
   else 
       t3=10^11;
   end
   if t3>=0 & t3<tmin2 &  ~(osumasivu2 ==3)
       uusiosumasivu2=3;
       tmin2=t3;
       uusiP2=P2+t3*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv2=vw-vwk;
   end
   %alasivu
   C=[-L,-K+r2];
   w=[1,0];
   wk=[-w(2),w(1)];
   alakerta=dot(v,wk);
   if alakerta^2>0
       t4=dot(C-A,wk)/alakerta;
   else 
       t4=10^11;
   end
   if t4>=0 & t4<tmin2 & ~(osumasivu2 ==4)
       uusiosumasivu2=4;
       tmin2=t4;
       uusiP2=P2+t4*v;
       vw=dot(v,w)/norm(w)^2*w;
       vwk=v-vw;
       uusiv2=vw-vwk;
   end

%pallojen törmäyshetki tcoll   
a=norm(v1-v2)^2;
b=2*dot(P1-P2,v1-v2);
c=norm(P1-P2)^2-(r1+r2)^2;
d=b^2-4*a*c;
if d>0
tcoll=(-b-sqrt(d))/(2*a);
else 
    tcoll=10^10;
end

%keskipisteiden paikat törmäyshetkellä
P1c=P1+tcoll*v1;
P2c=P2+tcoll*v2;
%nopeudet törmäyksen jälkeen
w=P2c-P1c;
v1w=dot(v1,w)/norm(w);
v1wk=v1-v1w*w/norm(w);
v2w=dot(v2,w)/norm(w);
v2wk=v2-v2w*w/norm(w);
u1=(m1-m2)/(m1+m2)*v1w+2*m2/(m1+m2)*v2w;
u1w=u1*w/norm(w);
u2=2*m1/(m1+m2)*v1w+(m2-m1)/(m1+m2)*v2w;
u2w=u2*w/norm(w);
v1c=u1w+v1wk;
v2c=u2w+v2wk;


if tcoll<0
    tcoll=10^10
end

tmin=min([tmin1,tmin2,tcoll])
trata=dt:dt:tmin;
P1ratax=[P1ratax,P1(1)+trata*v1(1)];
P1ratay=[P1ratay,P1(2)+trata*v1(2)];
P2ratax=[P2ratax,P2(1)+trata*v2(1)];
P2ratay=[P2ratay,P2(2)+trata*v2(2)];

if tmin==tmin1 %pallo1 osuu ensin seinään   
   P1=uusiP1;
   v1=uusiv1;
   osumasivu1=uusiosumasivu1;   
   P2=P2+tmin*v2;
   osumasivu2=0;
elseif tmin==tmin2 %pallo2 osuu ensin seinään 
   P2=uusiP2;
   v2=uusiv2;
   osumasivu2=uusiosumasivu2;
   P1=P1+tmin*v1;
   osumasivu1=0;
else %pallot törmäävät toisiinsa
   P1=P1c;
   v1=v1c;
   osumasivu1=5;
   P2=P2c;
   v2=v2c;
   osumasivu2=5;
end

    
end

% animaatio


c=cosd(0:5:360);
s=sind(0:5:360);
Nm=length(P1ratax);
   
for m=1:Nm
plot([L,L,-L,-L,L],[-K,K,K,-K,-K],'k','linewidth',2)
hold on
plot(P1ratax(m)+r1*c,P1ratay(m)+r1*s,'r','linewidth',2)
plot(P2ratax(m)+r2*c,P2ratay(m)+r2*s,'b','linewidth',2)
hold off
axis([-L-1,L+1,-K-1,K+1])
axis equal
pause(0.001)
end
