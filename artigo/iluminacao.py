import numpy as np, pandas as pd, onnxruntime as ort
from PIL import Image
M=np.array([0.485,0.456,0.406],np.float32); D=np.array([0.229,0.224,0.225],np.float32)
m=pd.read_csv('dados/manifesto.csv')
m=m[(m.split=='test')&m.consistente&(m.saturacao>=12)&m.raca.isin(['branca','parda','preta'])&(m.status=='ok')]
m=pd.concat([g.sample(min(len(g),500),random_state=0) for _,g in m.groupby('raca')]).reset_index(drop=True)
y=m.raca.map({'branca':0,'parda':1,'preta':2}).values
imgs=np.stack([np.asarray(Image.open('dados/'+a).convert('RGB').resize((160,160)),np.float32)/255 for a in m.arquivo])
s=ort.InferenceSession('modelos/aluno2/aluno2_w16.onnx')
lin=lambda x: np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
srgb=lambda x: np.where(x<=0.0031308,12.92*x,1.055*np.clip(x,0,None)**(1/2.4)-0.055).clip(0,1)
def luz(x,ganho): return srgb(lin(x)*np.array(ganho,np.float32))
# correções candidatas
def nada(x): return x
def mundo_cinza(x):  # gray-world no recorte inteiro (pele+fundo)
    l=lin(x); g=l.reshape(len(x),-1,3).mean(1); return srgb(l*(g.mean(1,keepdims=True)/g)[:,None,None,:])
def exposicao(x):   # só iguala o brilho médio ao típico do TSE
    l=lin(x); mu=l.mean((1,2,3),keepdims=True); return srgb(l*(ALVO/mu))
def ambos(x): return exposicao(mundo_cinza(x))
ALVO=lin(imgs).mean()
def prever(x):
    out=[]
    for i in range(0,len(x),100):
        z=s.run(None,{'imagem':((x[i:i+100]-M)/D).transpose(0,3,1,2).astype(np.float32)})[0][:,:3]
        e=np.exp(z-z.max(1,keepdims=True)); out.append(e/e.sum(1,keepdims=True))
    return np.concatenate(out)
cenas={'original':[1,1,1],'escuro (-1 stop)':[.5,.5,.5],'muito escuro (-2)':[.25,.25,.25],'claro (+1)':[2,2,2],
       'luz quente':[1.15,1,.7],'escuro+quente':[.55,.5,.35],'luz fria':[.85,1,1.25]}
for nome,f in [('sem correção',nada),('mundo cinza',mundo_cinza),('exposição',exposicao),('cor+exposição',ambos)]:
    print(f'\n== {nome} ==')
    base=None
    for c,gn in cenas.items():
        p=prever(f(luz(imgs,gn)))
        bal=np.mean([np.mean(p[y==k].argmax(1)==k) for k in range(3)])
        nota=p[:,1]+2*p[:,2]
        if base is None: base=nota
        br=y==0
        print(f'{c:18s} bal-acc {bal:.3f} | brancos→P(branca) {p[br,0].mean()*100:4.1f}% | desvio da nota vs original {np.abs(nota-base).mean():.2f}')
