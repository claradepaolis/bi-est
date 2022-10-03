function sig = check_conditioning(sig)
cond_min = 1000;
eps=0.001;
switch length(size(sig))
    case 4
        [C, dim, ~, K] = size(sig);
        for c=1:C
            for k=1:K
                s=reshape(sig(c,:,:,k),dim,dim);
                s_cond = cond(s);
                if s_cond > cond_min
                   sig(c,:,:,k)=recondition_sig(s, cond_min, dim, eps);
                end
            end
        end
        
    case 3
        [dim, ~, K] = size(sig);
        for k=1:K
            s=reshape(sig(:,:,k),dim,dim);
            s_cond = cond(s);
            if s_cond > cond_min
               sig(:,:,k)=recondition_sig(s, cond_min, dim, eps);
            end
        end
    case 2
        [dim, ~] = size(sig);
        s_cond = cond(sig);
        if s_cond > cond_min
           sig=recondition_sig(sig, cond_min, dim, eps);
        end
end
    
end

function sig=recondition_sig(sig_orig, cond_min, dim, eps)
    sig = sig_orig + (eps * eye(dim));
    sig_cond = cond(sig);
    while sig_cond > cond_min
        eps = eps * 2;
        sig = sig_orig + (eps * eye(dim));
        sig_cond = cond(sig);
    end
end