%% s.3-4
clear
v=[5,3]
norm(v)%||v|| = v:n pituus
sqrt(v(1)^2+v(2)^2)

%% s.7-8
clear
P=[-3,2]
Q=[5,7]
PQ=Q-P
norm(PQ) %||PQ|| = PQ:n pituus
%% s.11-12 napakoordinaatit
clear
v=[8,4]
theta=atan2(v(2),v(1)) %rad
thetad=atan2d(v(2),v(1)) %deg 
%% s.13-14
clear
u=[6,2]
v=[1,4]
u+v
3*v
%u*v% -> virheilmoitus
%% s.17, suoran parametrimuoto
clear
%close all
A=[4,3] %suoran piste
v=[2,5] %suuntavektori
t=1.2 
P=A+t*v
Av=A+1*v 
%suoran paatepisteet
tmin=-1
tmax=2
Pmin=A+tmin*v
Pmax=A+tmax*v
%%

plot([Pmin(1),Pmax(1)],[Pmin(2),Pmax(2)],'r','linewidth',1.5)
hold
p2=plot([A(1),Av(1)],[A(2),Av(2)],'b','linewidth',3) %vektori v
p1=plot(A(1),A(2),'k.','markersize',25) %piste A
p3=plot(P(1),P(2),'r.','markersize',20) %piste P
hold off
grid %taustaristikko
axis equal %akseleiden mittakaavat yhtasuuriksi
legend([p1,p2,p3],{'A','v','P'},'fontsize',12)
title(['t = ',num2str(t),', P = A+tv'])
%% s.22
clear 
v=[3,2]
v0=v/norm(v)
1.7*v0


%% s.23
clear
P=[2,4]
Q=[5,2]
PQ=Q-P
PQ0=PQ/norm(PQ) %yksikkovektori
pr=1.5 %||PR||
PR=pr*PQ0
R=P+PR
%%
plot([P(1),Q(1)],[P(2),Q(2)],'r','linewidth',1.5)
hold on
plot(P(1),P(2),'m.','markersize',20)
plot(Q(1),Q(2),'g.','markersize',20)
plot(R(1),R(2),'b.','markersize',20)
hold off
grid
axis equal
legend('PQ','P','Q','R') 
title(['||PR|| = ',num2str(pr)])
%% s.25
clear
L1=6
L2=5
H=13
F=20

P=[0,H]
A=[-L1,0]
B=[L2,0]
PA=A-P
u=PA/norm(PA)
PB=B-P
v=PB/norm(PB)
%%
D=u(1)*v(2)-u(2)*v(1)
fa=v(1)/D*F
fb=-u(1)/D*F
FA=fa*u
FB=fb*v

%tarkastus
FA+FB
[0,-F]
%%
L=max([L1,L2])+1
plot([-L,L],[0,0],'k')
hold on
plot([0,0],[0,H+1],'k')
plot([A(1),P(1),B(1)],[A(2),P(2),B(2)],'k','linewidth',1.2)
PFA=P+FA
p1=plot([P(1),PFA(1)],[P(2),PFA(2)],'r','linewidth',3)
PFB=P+FB
p2=plot([P(1),PFB(1)],[P(2),PFB(2)],'g','linewidth',3)
PFAFB=PFA+FB
plot([PFA(1),PFAFB(1)],[PFA(2),PFAFB(2)],'g','linewidth',1.2)
PFBFA=PFB+FA
plot([PFB(1),PFBFA(1)],[PFB(2),PFBFA(2)],'r','linewidth',1.2)
p3=plot([P(1),P(1)],[P(2),P(2)-F],'k','linewidth',3)
p4=plot(P(1),P(2),'k.','markersize',20)
p5=plot(A(1),A(2),'r.','markersize',20)
p6=plot(B(1),B(2),'g.','markersize',20)
hold off
grid
axis equal
legend([p1,p2,p3,p4,p5,p6],{'F_A','F_B','[0,-F]','P','A','B'},'fontsize',11)
title(['L_1 = ',num2str(L1),', L_2 = ',num2str(L2),', H = ',num2str(H),...
       ', F = ',num2str(F),...
        ': ||F_A|| = ',num2str(fa), ', ||F_B|| = ',num2str(fb)])

    
%% s.32, vektoreiden valinen kulma
clear
u=[3,2]
v=[-1,5]
cos_alfa=dot(u,v)/(norm(u)*norm(v))
alfa=acosd(cos_alfa)

 %% s.37
clear
v=[3,1]
vk=[-v(2),v(1)]
L=norm(v)
plot([0,v(1)],[0,v(2)],'b','linewidth',2)
hold
plot([0,vk(1)],[0,vk(2)],'g','linewidth',2)
plot([-L,L],[0,0],'k')
plot([0,0],[-L,L],'k')
hold off
grid
axis equal
legend({'v','vk'},'fontsize',13)
title(['v = [',num2str(v),'], vk = [',num2str(vk),']'])

%% s.39 suoran normaalimuoto ax+by=c
clear
a=2
b=5
c=23
n=[a,b] %suoran normaali
L=5 %suoran pituus = 2*L
%suoran piste
A=c/norm(n)*n/norm(n) %OA
%%
%suoran suuntainen (= n:aa vastaan kohtisuora) vektori, s.37
v=[-n(2),n(1)] 
%suoran paatepisteet
B=A+L*v/norm(v)
C=A-L*v/norm(v)

plot([B(1),C(1)],[B(2),C(2)],'g','linewidth',2)
hold
plot([0,n(1)],[0,n(2)],'k','linewidth',2)
plot(A(1),A(2),'b.','markersize',25)
plot(0,0,'k.','markersize',20)
plot([-L,L],[0,0],'k')
plot([0,0],[-L,L],'k')
hold off
grid
axis equal
legend({'ax+by=c','n=[a,b]','A=c/||n||*n^0'},'fontsize',12,...
        'location','southwest')
title(['a = ',num2str(a),', b = ',num2str(b),', c = ',num2str(c),...
       ', ||OA|| = ',num2str(norm(A))])
x=B(1)
y=B(2)
a*x+b*y
c
   
  
%% s. 45, onko piste P kolmion ABC sisa- vai ulkopuolella ? 
clear
close all
%kiertosuunta A->B->C myotapaivaan
A=[-2,1]
B=[5,4]
C=[9,0]
P=[1,2]

AB=B-A
ABk=[-AB(2),AB(1)]
BC=C-B
BCk=[-BC(2),BC(1)]
CA=A-C
CAk=[-CA(2),CA(1)]
AP=P-A
t1=dot(AP,ABk)
BP=P-B
t2=dot(BP,BCk)
CP=P-C
t3=dot(CP,CAk)
t=max([t1,t2,t3])
%%

plot([A(1),B(1),C(1),A(1)],[A(2),B(2),C(2),A(2)],'b','linewidth',2)
hold on
plot(P(1),P(2),'r.','markersize',20)
hold off
grid
axis equal
if t<=0
    title('sisalla')
else
    title('ulkona')
end

%% s.47 suorien leikkauspiste
clear
close all
%suora 1
A=[1,1] %piste
v=[6,2] %suuntavektori
%suora 2
C=[-2,6] %piste
w=[4,-2] %suuntavektori
%kohtisuorat
vk=[-v(2),v(1)]
wk=[-w(2),w(1)]
 
AC=C-A
t=dot(AC,wk)/dot(v,wk) %jos dot(v,wk)=0, niin suorat samansuuntaiset
P=A+t*v
CA=-AC
s=dot(CA,vk)/dot(w,vk)
C+s*w
%%
%tai suoraan funktiotiedostolla suorien_leikkauspiste_2D.m (oletushakemistossa)
[P,t,s]=suorien_leikkauspiste_2D(A,v,C,w)
%%
Amin=A-t*v
Amax=A+2*t*v
plot([Amin(1),Amax(1)],[Amin(2),Amax(2)],'r','linewidth',1.5)
hold on
Cmin=C-s*w
Cmax=C+2*s*w
plot([Cmin(1),Cmax(1)],[Cmin(2),Cmax(2)],'b','linewidth',1.5)
Av=A+v
p2=plot([A(1),Av(1)],[A(2),Av(2)],'g','linewidth',3)
p1=plot(A(1),A(2),'r.','markersize',20)
Cw=C+w
p4=plot([C(1),Cw(1)],[C(2),Cw(2)],'m','linewidth',3)
p3=plot(C(1),C(2),'b.','markersize',20)
p5=plot(P(1),P(2),'k.','markersize',20)
hold off
title(['P = A + t*v = C + s*w, t = ',num2str(t),', s = ',num2str(s)])
grid
axis equal
legend([p1,p2,p3,p4,p5],{'A','v','C','w','P'},'fontsize',10)
%% s. 51 pisteiden A,B,C kautta kulkeva ympyra
clear
A=[1,2]
B=[3,-1]
C=[2,4]
D=(B+C)/2
BC=C-B
BCk=[-BC(2),BC(1)]
E=(A+C)/2
AC=C-A
ACk=[-AC(2),AC(1)]
[P,t,s]=suorien_leikkauspiste_2D(D,BCk,E,ACk)
r=norm(A-P)
t=0:360;
xymp=P(1)+r*cosd(t);
yymp=P(2)+r*sind(t);

plot(xymp,yymp,'b','linewidth',1.5)
hold
plot(P(1),P(2),'b.','markersize',20)
plot([A(1),B(1),C(1)],[A(2),B(2),C(2)],'r.','markersize',20)
hold off
grid
axis equal
%% s.53 suoran ja ympyran lp
clear
close all
A=[4,4] %suoran piste
v=[2,1]  %suoran suuntavektori
P=[5,3] %ympyran kp
r=2 %ympyran sade

PA=A-P
a=norm(v)^2
b=2*dot(PA,v)
c=norm(PA)^2-r^2

%at^2+bt+c=0

D=b^2-4*a*c %neliojuuren sisus

if D>=0
    t1=(-b-sqrt(D))/(2*a)
    t2=(-b+sqrt(D))/(2*a)
    S=A+t1*v
    T=A+t2*v
end
Av=A+v
plot([A(1),Av(1)],[A(2),Av(2)],'k','linewidth',3)%v
hold on
plot(A(1),A(2),'b.','markersize',25)
k=0:360; %kiertokulma k=[0,1,2,...,360]
plot(P(1)+r*cosd(k),P(2)+r*sind(k),'r','linewidth',2) %ympyra
plot(P(1),P(2),'r.','markersize',20)

if D>=0    
    tmax=max([1,t1,t2])+1
    tmin=min([0,t1,t2])-1
    vp=A+tmin*v %suoran "vasen" piste
    op=A+tmax*v %suoran "oikea" piste
    plot([vp(1),op(1)],[vp(2),op(2)],'b','linewidth',1.5) %suora
    plot([S(1),T(1)],[S(2),T(2)],'g.','markersize',20)
    title(['t_1 = ',num2str(t1),', t_2 = ',num2str(t2)])
else
    title('eivat leikkaa')
end

hold off
grid
axis equal



%% s.63 komponentteihin jako
clear
close all
v=[-3,4]
u=[5,2]
vu=dot(v,u)/norm(u)^2*u
vuk=v-vu
%%
plot([0,u(1)],[0,u(2)],'r','linewidth',3)
hold
plot([0,v(1)],[0,v(2)],'g','linewidth',3)
plot([0,vu(1)],[0,vu(2)],'b','linewidth',3)
plot([vu(1),v(1)],[vu(2),v(2)],'m','linewidth',3)
hold off
grid
axis equal
legend({'u','v','v_u','v_{uk}'},'fontsize',12)
%% s.67 heijastuminen
clear
close all
A=[0,0]
v=[2,-3]
u=[4,-1]
vu=dot(v,u)/norm(u)^2*u
vuk=v-vu
w=vu-vuk
%%
n=[-u(2),u(1)]
Amin=A-u
Amax=A+2*u
plot([Amin(1),Amax(1)],[Amin(2),Amax(2)],'r','linewidth',1)
hold on
Au=A+u
p1=plot([A(1),Au(1)],[A(2),Au(2)],'k','linewidth',3)
Av=A-v
p2=plot([A(1),Av(1)],[A(2),Av(2)],'b','linewidth',3)
Aw=A+w
p3=plot([A(1),Aw(1)],[A(2),Aw(2)],'g','linewidth',3)
An=A+n
plot([A(1),An(1)],[A(2),An(2)],'k','linewidth',1)
hold off
grid
axis equal
legend([p1,p2,p3],{'u','v','w'},'fontsize',12)
%% s.71 nopeus ja kiihtyvyys
clear
v=[5,2]
a=[3,-4]
aTp=dot(a,v)/norm(v)
aT=aTp*v/norm(v)
vk=[-v(2),v(1)]
aNp=dot(a,vk)/norm(v)
aN=aNp*vk/norm(v)

plot([0,v(1)],[0,v(2)],'g','linewidth',3)
hold
plot([0,a(1)],[0,a(2)],'b','linewidth',3)
plot([0,aT(1)],[0,aT(2)],'m','linewidth',3)
plot([0,aN(1)],[0,aN(2)],'c','linewidth',3)
plot(0,0,'k.','markersize',30)
hold off
grid
axis equal
legend({'$\bf v$','$\bf a$','$\bf a_T$','$\bf a_N$'},'fontsize',14,'location','northwest','interpreter','latex')
title(['$a_T$ = ',num2str(aTp),', $a_N$ = ',num2str(aNp)],'fontsize',14,'interpreter','latex')
%% s.83 2D ristitulo
clear
u=[3,2]
v=[1,4]
uxv=u(1)*v(2)-u(2)*v(1)

%% s. 85 kolmion ala
clear
A=[1,2]
B=[4,4]
C=[0,6]

AB=B-A
AC=C-A
u=AB
v=AC
uxv=u(1)*v(2)-u(2)*v(1)
ala=1/2*(abs(uxv))

