"""
    Exemplo de aplicação do método de Runge-Kutta com 2 estágios (RK2) para resolver um sistema de 
    doze equações diferenciais ordinárias (EDOs), correspondentes a um problema de Gravitação entre
    três corpos de massas m1, m2 e m3, posições iniciais r1 = (x1, y1), r2 = (x2, y2) e r3 = (x3, y3)
    e momentos p1 = (px1, py1), p2 = (px2, py2) e p3 = (px3, py3).

    A modelagem é análoga à feita no exemplo 1, mas com 3 partículas no lugar de 2, então com 2 forças
    consideradas para cada partícula.

    Mantida essa ordem, geralmente se obterá um vetor 1x13 com as componentes:

        [t, x1, px1, y1, py1, x2, px2, y2, py2, x3, px3, y3, py3]

    Então é necessária uma maracutaia para acessar os pontos corretamente na hora de averiguar o
    resultado.
"""
from math import sqrt
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rk import RK

## CONDIÇÕES INICIAIS
# massas das partículas
m = [50, 30, 500]

# posições iniciais
posicoes_iniciais = [
    [-10, 10],
    [50, 10],
    [-10, -40]
]
# momentos iniciais
momentos_iniciais = [
    [100, 0],
    [-20, 30],
    [0, 0],
    # [-10, -2],
    # [-4, 8],
    # [40, -3]
]

# instante final
tf = 500

## FUNÇÕES AUXILIARES
# função de distância
r = lambda r1, r2: sqrt(sum((r1[i]-r2[i])**2 for i in range(len(r1))))

# funções (equações)
# equação do movimento de uma partícula (t é o instante, Ps é uma lista de partículas [xa, pax, ya, pay] para a = 1, ..., n, i = 1 == pax, i = 3 == pay)
fxa_ = lambda a, t, Ps: (1/m[a]) * Ps[4*a+1]
fya_ = lambda a, t, Ps: (1/m[a]) * Ps[4*a+3]
# equação do momento de uma partícula
fpxa_ = lambda a, t, Ps: m[a] * sum( m[b] * (Ps[4*b+0] - Ps[4*a+0])/r([Ps[4*a+0], Ps[4*a+2]], [Ps[4*b+0], Ps[4*b+2]])**3 for b in range(len(m)) if b != a)
fpya_ = lambda a, t, Ps: m[a] * sum( m[b] * (Ps[4*b+2] - Ps[4*a+2])/r([Ps[4*a+0], Ps[4*a+2]], [Ps[4*b+0], Ps[4*b+2]])**3 for b in range(len(m)) if b != a)

funcoes = [
    lambda t, *Ps: fxa_(0, t, Ps),
    lambda t, *Ps: fpxa_(0, t, Ps),
    lambda t, *Ps: fya_(0, t, Ps),
    lambda t, *Ps: fpya_(0, t, Ps),
    lambda t, *Ps: fxa_(1, t, Ps),
    lambda t, *Ps: fpxa_(1, t, Ps),
    lambda t, *Ps: fya_(1, t, Ps),
    lambda t, *Ps: fpya_(1, t, Ps),
    lambda t, *Ps: fxa_(2, t, Ps),
    lambda t, *Ps: fpxa_(2, t, Ps),
    lambda t, *Ps: fya_(2, t, Ps),
    lambda t, *Ps: fpya_(2, t, Ps),
]

## CONFIGURAÇÕES DO MÉTODO DE RUNGE-KUTTA
a = [[0, 0], [2/3, 0]]
b = [1/4, 3/4]
R = 2
h = 0.025
print(f"Quantidade de aplicações: {tf/h}")
y0 = [] # condições iniciais
for i in range(len(m)):
    y0.append(posicoes_iniciais[i][0])
    y0.append(momentos_iniciais[i][0])
    y0.append(posicoes_iniciais[i][1])
    y0.append(momentos_iniciais[i][1])
## APLICAÇÃO DO MÉTODO
rk = RK(
    f = funcoes,
    t0 = 0,
    y0 = y0,
    h = h,
    R = R,
    a = a,
    b = b
)

tempo = time()
# aplica o método
pontos = rk.aplicar(tf)
print(f'Tempo levado: {time() - tempo}')

## VISUALIZAÇÃO
# separa os pontos
pontos = list(zip(*pontos))
eixo_t = pontos[0]
eixos_y = pontos[1:]
# saltos
ph = 10
eixos_x = [pontos[4*i+1][::ph] for i in range(len(m))]
eixos_y = [pontos[4*i+3][::ph] for i in range(len(m))]
print(f"Quantidade de frames: {len(eixos_x[0])}")

# matplotlib
fig = plt.figure()
eixos = plt.axes(xlim=(-100, 100), ylim=(-100, 100))

pontos_plt = [eixos.scatter([], []) for i in range(len(m))]

def animar (frame:int)->list:
    """
        Função de animação para o matplotlib.animation.

        Parâmetros
        ----------
        frame : int
            Frame atual.
        
        Retorna
        -------
        ponto1, ponto2, ..., pontoN : list de pontos
            Lista de pontos no frame.
    """
    for i, ponto in enumerate(pontos_plt):
        ponto.set_offsets([eixos_x[i][frame], eixos_y[i][frame]])
    return pontos_plt

anim = animation.FuncAnimation(fig, animar, frames=len(eixos_x[0]), interval=1, blit=True)
plt.legend()
plt.show()