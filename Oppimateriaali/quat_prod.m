function QR=quat_prod(Q,R)
%QR on kvaternioiden Q ja R tulo
q0=Q(1);
q=Q(2:4);
r0=R(1);
r=R(2:4);
QR=[q0*r0-dot(q,r),q0*r+r0*q+cross(q,r)];
    