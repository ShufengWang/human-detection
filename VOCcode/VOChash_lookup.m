function ind = VOChash_lookup(hash,s)

hsize=numel(hash.key);
h=mod(str2double([s(end-4:end)]),hsize)+1;
ind=hash.val{h}(strmatch(s,hash.key{h},'exact'));
