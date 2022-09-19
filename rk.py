class RK:
    """
        Método de Runge-Kutta com R estágios para resolver sistemas de n EDOs a partir
        de uma condição inicial (Problema de Cauchy).
    """
    def __init__ (self, f: list, t0: float, y0: list, h:float, R:int, a:list, b:list):
        """
            Sistema de EDOs descritas em `f` com as condições iniciais `t0` e `y0`.

            Parâmetros
            ----------
            f : list
                Lista de funções `(f1(t, x1, ..., xn), ..., fn(t, x1, ..., xn))`.
            t0 : float
                Instante inicial.
            y0 : list
                Condições iniciais `(x1(t0), x2(t0), ..., xn(t0))`.
            h : float
                Tamanho do passo `(x1 - x0 = h)`.
            R : int
                Quantidade de estágios.
            a : list
                Matriz de Runge-Kutta.
            b : list
                Matriz de coeficientes de peso.
        """
        self.R = R # quantidade de estágios
        self.n = len(f) # dimensão
        self.f = f # vetor de funções f = (f1, f2, ..., fn)
        self.t0 = t0 # instante inicial
        self.y0 = y0 # condições iniciais (x1(t0), x2(t0), ..., xn(t0))
        self.h = h # tamanho do passo
        self.a = a # matriz de Runge-Kutta (n x n)
        self.b = b # matriz de coeficientes de peso (1 x n)
        self.c = [ sum(ai) for ai in self.a] # matriz (n x 1)

        # margem considerada para o tf
        self.qntdCasasArredondamento = 10

    def phi (self, f:'function', t:float, y:list)->float:
        """
            Função discreta de `f`.

            Parâmetros
            ----------
            f : function
                Função que se está considerando.
            t : float
                Instante.
            y : list
                Lista de condições atuais.
            
            Retorna
            -------
            sum (b_r k_r) : float
                Média ponderada por `b` dos `kappa`.
        """
        return sum (
            self.b[r] * self.kappa(f, t, y, r) for r in range(self.R)
        )

    def kappa (self, f: 'function', t: float, y: list, r: int)->float:
        """
            Estágio `r` de uma determinada aplicação do método.

            Parâmetros
            ----------
            f : function
                Função que se está considerando.
            t : float
                Instante.
            y : list
                Lista de condições atuais.
            r : int
                Estágio.

            Retorna
            -------
            f(t + h c_r, y_i + h sum a_rs k_s) : float
                Aplicação do instante e de um pequeno acréscimo nas condições.
        """
        instante = t + self.h * self.c[r] # t + h c_r
        termos_y = [
            y[i] + self.h * sum ( self.a[r][s] * self.kappa(f, t, y, s) for s in range(r-1) )
            for i in range(self.n)
        ]

        return f(instante, *termos_y)

    def yk1 (self, tk: float, yk: list)->list:
        """
            Passo k + 1 a partir do passo k.

            Parâmetros
            ----------
            tk : float
                Instante anterior.
            yk : list
                Condições do estágio anterior.

            Retorna
            -------
            tk1, yk1 : float, list
                Novas condições do passo k + 1.
        """
        tk1 = tk + self.h # acréscimo no instante
        yk1 = [
            yk[i] + self.h * self.phi(self.f[i], tk, yk) for i in range(self.n)
        ]
        return tk1, yk1

    def aplicar (self, tf: float)->list:
        """
            Aplica o método de Runge-Kutta com R estágios (conforme instanciado)
            até no intervalo [t0, tf].

            Parâmetros
            ----------
            tf : float
                Instante final desejado.

            Retorna
            -------
            pontos : list
                Lista com os valores das equações em cada instante.
        """
        pontos = [ [self.t0, *self.y0] ]

        tk = self.t0 # instante k
        yk = self.y0 # condições k

        while True:
            tk, yk = self.yk1(tk, yk) # calcula
            pontos += [[tk, *yk]] # salva    
            if round(tk, self.qntdCasasArredondamento) >= tf: break
        return pontos