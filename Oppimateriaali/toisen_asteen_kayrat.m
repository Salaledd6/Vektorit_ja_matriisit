clear
close
%toisen asteen käyrä
%Ax^2+Bxy+Cy^2+Dx+Ey=F
A=1
B=4
C=4
D=-4
E=-3
F=2



r=4 %u- ja v-akseleiden pituus = 2r 


%kiertokulma th 
if A==C & B==0
    th=0
else
    th=1/2*atan2d(B,A-C)
end

c=cosd(th);
s=sind(th);

%yhtälö kierretyssä uv-koordinaatistossa
%alfa*u^2+beta*u*v+gamma*v^2+delta*u+epsilon*v=F
alfa=A*c^2+B*c*s+C*s^2
beta=-2*A*c*s+B*(c^2-s^2)+2*C*c*s %=0
gamma=A*s^2-B*c*s+C*c^2
delta=D*c+E*s
epsilon=-D*s+E*c

%tapaus 1
if alfa~=0 & gamma~=0
    u0=-delta/(2*alfa) 
    v0=-epsilon/(2*gamma)
    k=F+alfa*(delta/(2*alfa))^2+gamma*(epsilon/(2*gamma))^2
    
    if k/alfa>0 & k/gamma>0 %ellipsi
        a=sqrt(k/alfa);
        b=sqrt(k/gamma);
        t=linspace(0,360,100);
        u=u0+a*cosd(t);
        v=v0+b*sind(t);
        x=c*u-s*v;
        y=s*u+c*v;
        x0=c*u0-s*v0;
        y0=s*u0+c*v0;
        plot(x,y,'b','linewidth',2)
        hold;
        plot([0,0+c],[0,0+s],'r','linewidth',3)
       plot([0,0-s],[0,0+c],'g','linewidth',3)
       p1=plot([0-r*c,0+r*c],[0-r*s,0+r*s],'r','linewidth',1);
       p2=plot([0+r*s,0-r*s],[0-r*c,0+r*c],'g','linewidth',1);
       p3=plot(x0,y0,'k.','markersize',20);
       hold off     
       grid
       axis equal
       title({['A = ',num2str(A),', B = ',num2str(B),', C = ',num2str(C),...
                ', D = ',num2str(D),', E = ',num2str(E),', F = ',num2str(F)];,...
               ['\theta = ',num2str(th,4),'^\circ, u_0 = ',num2str(u0,3),...
                ', v_0 = ',num2str(v0,3), ', x_0 = ',num2str(x0,3),...
                ', y_0 = ',num2str(y0,3),',   a = ',num2str(a,3),...
                ', b = ',num2str(b,3)]},...
                'fontsize',12)
       legend([p1,p2,p3],{'u','v','P_0'},'fontsize',12)
        
    elseif k/alfa>0 & k/gamma<0 %hyperbeli, aukeaa u-akselin suuntaan
        a=sqrt(k/alfa);
        b=sqrt(-k/gamma);
        tmax=acosh(r/a);
        %a*cosh(t)=r->t=acosh(r/a)
        t=linspace(-tmax,tmax,100);
        u1=u0+a*cosh(t);
        u2=u0-a*cosh(t);
        v=v0+b*sinh(t);
        x1=c*u1-s*v;
        y1=s*u1+c*v;
        x2=c*u2-s*v;
        y2=s*u2+c*v;        
        x0=c*u0-s*v0;
        y0=s*u0+c*v0;
        plot(x1,y1,'b',x2,y2,'b','linewidth',2)
        hold
        plot([0,0+c],[0,0+s],'r','linewidth',3)
       plot([0,0-s],[0,0+c],'g','linewidth',3)
       p1=plot([0-r*c,0+r*c],[0-r*s,0+r*s],'r','linewidth',1);
       p2=plot([0+r*s,0-r*s],[0-r*c,0+r*c],'g','linewidth',1);
       p3=plot(x0,y0,'k.','markersize',20);
       hold off     
       grid
       axis equal
       title({['A = ',num2str(A),', B = ',num2str(B),', C = ',num2str(C),...
                ', D = ',num2str(D),', E = ',num2str(E),', F = ',num2str(F)];,...
               ['\theta = ',num2str(th,4),'^\circ, u_0 = ',num2str(u0,3),...
                ', v_0 = ',num2str(v0,3), ', x_0 = ',num2str(x0,3),...
                ', y_0 = ',num2str(y0,3),',   a = ',num2str(a,3),...
                ', b = ',num2str(b,3)]},...
                'fontsize',12)
       legend([p1,p2,p3],{'u','v','P_0'},'fontsize',12)
        
        
        %tarkastusta varten
        x=x1;
        y=y1;
        
    elseif k/alfa<0 & k/gamma>0 %hyperbeli, aukeaa v-akselin suuntaan
        a=sqrt(-k/alfa);
        b=sqrt(k/gamma);
        tmax=acosh(r/a);
        %a*cosh(t)=r->t=acosh(r/a)
        t=linspace(-tmax,tmax,100);
        u=u0+a*sinh(t);
        v1=v0+b*cosh(t);
        v2=v0-b*cosh(t);
        x1=c*u-s*v1;
        y1=s*u+c*v1;
        x2=c*u-s*v2;
        y2=s*u+c*v2;
        x0=c*u0-s*v0;
        y0=s*u0+c*v0;
        plot(x1,y1,'b',x2,y2,'b','linewidth',2)
        hold
        plot([0,0+c],[0,0+s],'r','linewidth',3)
       plot([0,0-s],[0,0+c],'g','linewidth',3)
       p1=plot([0-r*c,0+r*c],[0-r*s,0+r*s],'r','linewidth',1);
       p2=plot([0+r*s,0-r*s],[0-r*c,0+r*c],'g','linewidth',1);
       p3=plot(x0,y0,'k.','markersize',20);
       hold off     
       grid
       axis equal
       title({['A = ',num2str(A),', B = ',num2str(B),', C = ',num2str(C),...
                ', D = ',num2str(D),', E = ',num2str(E),', F = ',num2str(F)];,...
               ['\theta = ',num2str(th,4),'^\circ, u_0 = ',num2str(u0,3),...
                ', v_0 = ',num2str(v0,3), ', x_0 = ',num2str(x0,3),...
                ', y_0 = ',num2str(y0,3),',   a = ',num2str(a,3),...
                ', b = ',num2str(b,3)]},...
                'fontsize',12)
       legend([p1,p2,p3],{'u','v','P_0'},'fontsize',12)
        
        %tarkastusta varten
        x=x1;
        y=y1;
        
    end
%tapaus 2    
elseif alfa~=0 & gamma==0 & epsilon~=0 %paraabeli, aukeaa v-akselin suuntaan
       a=-alfa/epsilon;
       u0=-delta/(2*alfa);
       v0=F/epsilon+alfa/epsilon*(delta/(2*alfa))^2;
       %v=a*(u-u0)^2+v0=v0+r->u-u0=sqrt(r/a)
       umin=u0-sqrt(r/abs(a));
       umax=u0+sqrt(r/abs(a));
       u=linspace(umin,umax,100);
       v=a*(u-u0).^2+v0;
       x=c*u-s*v;
       y=s*u+c*v;       
       x0=c*u0-s*v0;
       y0=s*u0+c*v0;
       plot(x,y,'b','linewidth',2)
       hold
       plot([0,0+c],[0,0+s],'r','linewidth',3)
       plot([0,0-s],[0,0+c],'g','linewidth',3)
       p1=plot([0-r*c,0+r*c],[0-r*s,0+r*s],'r','linewidth',1);
       p2=plot([0+r*s,0-r*s],[0-r*c,0+r*c],'g','linewidth',1);
       p3=plot(x0,y0,'k.','markersize',20);
       hold off     
       grid
       axis equal
       title({['A = ',num2str(A),', B = ',num2str(B),', C = ',num2str(C),...
                ', D = ',num2str(D),', E = ',num2str(E),', F = ',num2str(F)];,...
               ['\theta = ',num2str(th,4),'^\circ, u_0 = ',num2str(u0,3),...
                ', v_0 = ',num2str(v0,3), ', x_0 = ',num2str(x0,3),...
                ', y_0 = ',num2str(y0,3),',   a = ',num2str(a,3)]},...
                'fontsize',12)
       legend([p1,p2,p3],{'u','v','P_0'},'fontsize',12)
       
elseif alfa==0 & gamma~=0 & delta~=0 %paraabeli, aukeaa u-akselin suuntaan
       a=-gamma/delta;
       v0=-epsilon/(2*gamma);
       u0=F/delta+gamma/delta*(epsilon/(2*gamma))^2;
       %u=a*(v-v0)^2+u0=u0+r->v-v0=sqrt(r/a)
       vmin=v0-sqrt(r/abs(a));
       vmax=v0+sqrt(r/abs(a));
       v=linspace(vmin,vmax,100);
       u=a*(v-v0).^2+u0;
       x=c*u-s*v;
       y=s*u+c*v;
       x0=c*u0-s*v0;
       y0=s*u0+c*v0;
       plot(x,y,'b','linewidth',2)
       hold
       plot([0,0+c],[0,0+s],'r','linewidth',3)
       plot([0,0-s],[0,0+c],'g','linewidth',3)
       p1=plot([0-r*c,0+r*c],[0-r*s,0+r*s],'r','linewidth',1);
       p2=plot([0+r*s,0-r*s],[0-r*c,0+r*c],'g','linewidth',1);
       p3=plot(x0,y0,'k.','markersize',20);
       hold off     
       grid
       axis equal
       title({['A = ',num2str(A),', B = ',num2str(B),', C = ',num2str(C),...
                ', D = ',num2str(D),', E = ',num2str(E),', F = ',num2str(F)];,...
               ['\theta = ',num2str(th,4),'^\circ, u_0 = ',num2str(u0,3),...
                ', v_0 = ',num2str(v0,3), ', x_0 = ',num2str(x0,3),...
                ', y_0 = ',num2str(y0,3),',   a = ',num2str(a,3)]},...
                'fontsize',12)
       legend([p1,p2,p3],{'u','v','P_0'},'fontsize',12)
end

%tarkastus
k=61
A*x(k)^2+B*x(k)*y(k)+C*y(k)^2+D*x(k)+E*y(k)
F
    