# -*- coding: utf-8 -*-
"""
Created on Sat Jul  18 16:14:24 2020
@author: Rodrigo S. de Oliveira
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import integrate
from scipy import interpolate

def delta(m,n): #função delta de Kronecker
    if m==n:
        funcao = int(1)
    else:
        funcao = int(0)
    return funcao
'''
A função operadores(momento_angular_total) função calcula as matrizes Jx, Jy, Jz para um
determinado valor de momento angular total atribuído a variável momento_angular_total.
'''
def operadores(momento_angular_total):
    if momento_angular_total == 0:
        operador_momento_total = 'Não existe operador momento angular total definido para J = 0.'
        print('-> -> -> ERRO NA FUNÇÃO operadores(momento_angular_total)!')
    else:
        numero_de_linhas = int(2*(abs(momento_angular_total)) + 1)
        numero_de_colunas = int(2*(abs(momento_angular_total)) + 1)
        lam = (abs(momento_angular_total))*((abs(momento_angular_total)) + 1) 
        #lam é o valor J(J+1) referente ao autovalor do operador J_quadrado.
        Operador_Jz = np.zeros((numero_de_linhas,numero_de_colunas))
        Operador_J_mais = np.zeros((numero_de_linhas,numero_de_colunas))
        Operador_J_menos = np.zeros((numero_de_linhas,numero_de_colunas))
        for i in range(numero_de_linhas):
            m = (-1)*(abs(momento_angular_total)) + i
            for k in range(numero_de_colunas):
                n = (-1)*(abs(momento_angular_total)) + k
                Operador_Jz[i,k] = (m)*(delta(i,k))
                Operador_J_menos[i,k]= ((lam - (n)*(n - 1))**(0.5))*(delta(i,k-1))
                Operador_J_mais[i,k] = ((lam - (n)*(n + 1))**(0.5))*(delta(i,k+1))
        Operador_Jx = (Operador_J_mais + Operador_J_menos)/2
        Operador_Jy = (1/(2j))*(Operador_J_mais - Operador_J_menos)
        operador_momento_total = [Operador_Jx,Operador_Jy,Operador_Jz]
    return operador_momento_total
'''
A função funcao_particao(ener,Temp) calcula a função partição da mecânica estatística 
"ener" é a matriz com os autovalores de energia e "Temp" é a temperatura.
'''
def funcao_particao(ener,Temp): 
    C_B = 0.086 # Constante de Boltzmann em meV
    if Temp == 0:
        z = 'beta  = 1/((C_B)*(Temp)) não pode receber temperatura igual à zero'
        print('-> -> -> ERRO NA FUNÇÃO funcao_particao(ener,Temp)!')
    else:
        beta  = 1/((C_B)*(Temp))
        z = 0
        n = len(ener)
        for i in range(n):
            fator = np.exp((-beta)*(ener[i]))
            z += fator   
    return z
'''
A função energia_livre(energia,temp) calcula a energia livre para uma rede com várias sub-redes
'''
def energia_livre(energia,temp):
    C_B = 0.086 # Constante de Boltzmann em meV
    sub_nets,ene_bolt, F = len(energia), (C_B)*(temp), 0
    for i in range(sub_nets):
        Z = funcao_particao(energia[i],temp)
        F += (ene_bolt)*(np.log(Z))
    return F
'''
A função campo_efetivo(M_0,lande,lam,field, fi, theta) campo efetivo para cada sub-rede magnética
salvando em uma lista [bef1,bef2,...,befn], em que befi = [befxi, befyi,befzi] ou um array
M_0 é o vetor magnetização inicial de cada sub-rede (M_0 = [[m1x,m1y,m1z],[m2x,m2y,m2z],...[mnx,mny,mnz]] ou um array)
lande é do tipo [g_1, g_2,...gn], em que g_n é o fator de Landè da sub-rede 'n'.
lam é uma matriz n x n, em que os elementos da diagonal principal são os parâmetros de troca em meV
da sub-rede 'i' com ela mesma, e os outros elementos da matriz são os parâmetros de troca em meV
da sub-rede 'i' com as outras sub-redes (lam = [[L11,L12,...,L1n],[L21,L22,...,L2n],...[Ln1,Ln2,...,Lnn]] ou um array)
field é o campo magnético externo em Tesla
theta é o ângulo com o eixo 'x' da projeção do vetor campo magnético externo no plano xy
fi é o ângulo com o eixo 'z' do vetor campo magnético externo
'''
def campo_efetivo(M_0,lande,lam,campo):
    m_B = 0.0578838 # Magneton de Bohr em meV
    Bx = campo[0]
    By = campo[1]
    Bz = campo[2]
    Alpha = (m_B)*lande
    campo_efetivo = []
    for i in range (len(lande)):
        Bef_x = (Alpha[i])*(Bx)
        Bef_y = (Alpha[i])*(By)
        Bef_z = (Alpha[i])*(Bz)
        for j in range (len(lande)):
            Bef_x += ((lam[i,j])/(Alpha[j]))*(M_0[j][0]) 
            Bef_y += ((lam[i,j])/(Alpha[j]))*(M_0[j][1])
            Bef_z += ((lam[i,j])/(Alpha[j]))*(M_0[j][2])
            Bef = np.array([Bef_x,Bef_y,Bef_z])
        campo_efetivo.append(Bef)
    return campo_efetivo
'''
A função hamiltoniano(operador_spin,Bef,campocristalino) calula as energias e as autofunções do operador hamiltoniano
de cada sub-rede. Cada elemento da lista 'eigenvalues' contém os valores de energia da sub-rede 'i'. 
Cada elemento da lista 'eigenvectors' contém os autovetores de energia da sub-rede 'i'
operador_spin é o operador de spin de cada sub-rede. Ela deve ser uma lista do tipo
operador_spin = [operadores(J1),operadores(J2),...,operadores(Jn)].
Bef é o campo efetivo da sub-rede calculado pela função campo_efetivo(M_0,lande,lam,field, fi, theta)
campocristalino é o campo cristalino de cada sub-rede.Ela deve ser uma lista do tipo
operador_spin = [Hc1,Hc2,...,Hcn], em que Hci é uma matriz da ordem (2*(Ji) + 1)
'''
def hamiltoniano(operador_spin,Bef,campocristalino):
    eigenvalues = []
    eigenvectors = []
    for i in range(len(operador_spin)):
        H = - Bef[i][0]*(operador_spin[i][0]) - Bef[i][1]*(operador_spin[i][1]) - Bef[i][2]*(operador_spin[i][2]) + campocristalino[i]
        autovalores, autovetores = linalg.eigh(H)
        eigenvalues.append(autovalores)
        eigenvectors.append(autovetores)
    return eigenvalues, eigenvectors
'''
A função anisotropia_de_forma(d,spin) calcula o hamiltoniano de campo cristalino para uma anisotropia de forma.
D é o parâmetro de anisotropia e spin é o valor do momento angular total para esse sistema.
'''
def anisotropia_de_forma(d,spin):
    if d ==0:
        Hc = 0
    else:
        J = operadores(spin)
        Jz2 = J[2]@J[2]
        Hc = -d*Jz2
    return Hc
'''
A função campo_cristalino_cubico(constantes,spin) calcula o campo cristalino
de geometria cúbica na notação de Lea-Leask-Wolf, para o composto de momento
angular total de valor igual a variável spin
constantes = [W, F4,F6, xsi] são as contantes definadas no trabalho Lea-Leask-Wolf
'''
def campo_cristalino_cubico(constantes,spin):
    W, F4, F6, xsi = constantes[0],constantes[1],constantes[2], constantes[3]
    if W==0:
        Hc = 0
    else:
        momento_total = operadores(spin)
        Jx,Jy,Jz = momento_total[0], momento_total[1], momento_total[2]
        Jmais, Jmenos, I = Jx + (1j)*Jy, Jx - (1j)*Jy, np.identity(len(Jx))
        J, A = spin*I,(spin + 1)*I #fractional_matrix_power((Jx@Jx + Jy@Jy + Jz@Jz),0.5)
        J2, J3 = J@J, J@J@J
        A2, A3 = (A)@(A), (A)@(A)@(A)
        Jz2, Jz4, Jz6 = Jz@Jz, Jz@Jz@Jz@Jz, Jz@Jz@Jz@Jz@Jz@Jz
        Jmais4, Jmenos4 = Jmais@Jmais@Jmais@Jmais, Jmenos@Jmenos@Jmenos@Jmenos
        x1 = (35)*(Jz4)
        x2 = -(((30)*(J@A) - 25*I))@Jz2
        x3 = (-6)*(J@A) 
        x4 = (3)*(J2@A2)
        O40 = x1 + x2 + x3 + x4
        O44 = (1/2)*(Jmais4 + Jmenos4)
        O4 = O40 + (5)*O44
        y1 = 231*Jz6 - 315*J@A@Jz4 + 735*Jz4 
        y2 = 105*J2@A2@Jz2 - 525*J@A@Jz2 + 294*Jz2
        y3 = -5*J3@A3 + 40*J2@A2 - 60*J@A
        O60 = y1 + y2 + y3
        z1 = (11*Jz2 - J@A - 38*I)
        z2 = (z1)@(Jmais4 + Jmenos4)
        z3 = (Jmais4 + Jmenos4)@(z1)
        O64 = (1/4)*(z2 + z3)
        O6 = O60 + (-21)*O64
        psi = (1-abs(xsi))
        Hc = W*((xsi)*(O4/F4) + (psi)*(O6/F6))
    return Hc
'''
A função campo_cristalino_hexagonal(constantes,spin) calcula
a hamiltoniana de campo cristalino para uma rede de momento magnético total
dos ions igual a spin. constantes = [B02,B04,B06,B66] são as constantes de campo cristalino.
'''
def campo_cristalino_hexagonal(constantes,spin):
    B02, B04,B06, B66 = constantes[0],constantes[1],constantes[2],constantes[3]
    momento_total = operadores(spin)
    Jx,Jy,Jz = momento_total[0], momento_total[1], momento_total[2]
    Jmais, Jmenos, I = Jx + (1j)*Jy, Jx - (1j)*Jy, np.identity(len(Jx))
    J, A = spin*I,(spin + 1)*I #fractional_matrix_power((Jx@Jx + Jy@Jy + Jz@Jz),0.5)
    J2, J3 = J@J, J@J@J
    A2, A3 = (A)@(A), (A)@(A)@(A)
    Jz2, Jz4, Jz6 = Jz@Jz, Jz@Jz@Jz@Jz, Jz@Jz@Jz@Jz@Jz@Jz
    Jmais6, Jmenos6 = Jmais@Jmais@Jmais@Jmais@Jmais@Jmais, Jmenos@Jmenos@Jmenos@Jmenos@Jmenos@Jmenos
    O20 = (3)*Jz2 - J@A
    x1 = (35)*(Jz4)
    x2 = -(((30)*(J@A) - 25*I))@Jz2
    x3 = (-6)*(J@A) 
    x4 = (3)*(J2@A2)
    O40 = x1 + x2 + x3 + x4
    y1 = 231*Jz6 - 315*J@A@Jz4 + 735*Jz4 
    y2 = 105*J2@A2@Jz2 - 525*J@A@Jz2 + 294*Jz2
    y3 = -5*J3@A3 + 40*J2@A2 - 60*J@A
    O60 = y1 + y2 + y3
    O66 = (1/2)*(Jmais6 + Jmenos6)
    Hc = B02*O20 + B04*O40 + B06*O60 + B66*O66
    return Hc
'''
A função magnetizacao(lande,operador_spin,energia,psi,Temp) calcula a magnetização
nas direções x,y e z para a sub-rede magnética 'i' em uma temperatura fixa.
lande é o fator de Landè das sub-redes ([g_1, g_2,...gn]).
operador_spin = [operadores(J1),operadores(J2),...,operadores(Jn)], 
em que Ji é o valor absoluto do momento angular total da sub-rede 'i'
ener e psi é um array com os autovalores e autovetores do Hamiltoniano que descreve a sub-rede "i"
No array ener, cada item é um autovalor
No array psi, cada coluna é um autovetor referente ao autovalor da posição 'i'.
'''
def magnetizacao(lande,operador_spin,energia,psi,Temp): # g é o fator de Landè da sub-rede magnética
    m_B = 0.0578838 # Magneton de Bohr em meV 
    C_B = 0.086 # Constante de Boltzmann em meV
    if Temp == 0:
        MAG = 'beta  = 1/((C_B)*(Temp)) não pode receber temperatura igual à zero'
        print('-> -> -> ERRO NA FUNÇÃO magnetizacao(lande,operador_spin,energia,psi,Temp)!')
    else:
        M = []
        beta  = 1/((C_B)*(Temp))
        for j in range(len(lande)):
            Z = funcao_particao(energia[j],Temp)
            n = len(energia[j])
            soma_x, soma_y, soma_z = 0, 0, 0
            for i in range(n):
                v = psi[j][:,i]
                v_estr = (v.transpose()).conj()
                k_x = (v_estr)@(operador_spin[j][0])@(v)
                k_y = (v_estr)@(operador_spin[j][1])@(v)
                k_z = (v_estr)@(operador_spin[j][2])@(v)
                soma_x += (k_x)*(np.exp((-beta)*(energia[j][i])))
                soma_y += (k_y)*(np.exp((-beta)*(energia[j][i])))
                soma_z += (k_z)*(np.exp((-beta)*(energia[j][i])))
            magx = (lande[j])*(m_B)*((soma_x)/Z)
            magy = (lande[j])*(m_B)*((soma_y)/Z)
            magz = (lande[j])*(m_B)*((soma_z)/Z)
            mag = np.array([magx,magy,magz])
            M.append(mag)
        MAG = np.array(M)
    return MAG
'''
A função entropia_magnetica(energia,Temperatura) calcula a entropia magnética
de cada sub-rede em uma temperatura fixa
'''
def entropia_magnetica(energia,Temperatura):
    C_B, R = 0.086, 8.3144621  # Constante de Boltzmann em meV e a constante universal dos gases perfeitos(J/mol.K)
    if Temperatura == 0:
        S = 'beta  = 1/((C_B)*(Temp)) não pode receber temperatura igual à zero'
        print('-> -> -> ERRO NA FUNÇÃO entropia_magnetica(energia,Temperatura)!')
    else:
        beta  = 1/((C_B)*(Temperatura)) #expoente do Fator de Boltzmann
        S = [] #Lista para guardar os valores de entropia para cada sub-rede
        for energy in energia: 
            n = len(energy) #Lê o comprimento do array energy
            Z = funcao_particao(energy,Temperatura) #Calcula a função partição para a sub-rede que está no loop
            soma = 0
            for i in range(n):
                fator = np.exp((-beta)*(energy[i])) #Fator de Boltzmann
                soma += energy[i]*fator #O for em 'i' soma aqui cada valor de energia ao fator de Boltmann
                epsolon = soma/Z #dividi a soma pela função partição
                Entropia = R*(np.log(Z) + (beta)*(epsolon))#Calcula a entropia a sub-rede 'i'
            S.append(Entropia) #Salva o valor de entropia da sub-rede 'i'
    return S
'''
A função autoconsistencia(valores_entrada) calcula os valores de magnetização, entropia e energia livre.
valores_entrada = [J,Hc,g,lam,P,M_0,field,fi,theta,temp] é uma lista com esse conjunto de valores
J é uma lista com o momento angular total para cada sub-rede (J=[operadores(J1),operadores(J2),...,operadores(Jn)]).
Hc é uma lista que contém o campo cristalino para cada sub-rede (Hc = [Hc1, Hc2,...,Hcn]).
g é um array unidimensional cujos valores são os fatores de landé para cada sub-rede ([g_1, g_2,...gn]).
lam são os parâmetros de troca em meV. Ele é um array em forma do tipo (lam = [[L11,L12,...,L1n],[L21,L22,...,L2n],...[Ln1,Ln2,...,Lnn]]).
M_0 é chute inicial da magnetização de cada sub-rede (M_0 = [[m1x,m1y,m1z],[m2x,m2y,m2z],...[mnx,mny,mnz]]).
field é um float que representa o módulo do campo magnético aplicado em Tesla.
fi é um float que representa o ângulo que o campo magnético formna com o eixo z.
theta é um float que representa o ângulo que o campo magnético forma com o eixo x no plano xy.
temp é um float que representa a temperatura em Kelvin.
'''
def autoconsistencia(valores_entrada):
    J,Hc,g = valores_entrada[0],valores_entrada[1],valores_entrada[2]
    lam,P,M_0 = valores_entrada[3], valores_entrada[4], valores_entrada[5]
    field,fi,theta,temp = valores_entrada[6], valores_entrada[7], valores_entrada[8], valores_entrada[9]
    m_b = 0.0578838 # magneton de Bohr(meV/T^2).
    vec_u = np.array([(np.sin(fi))*(np.cos(theta)),(np.sin(fi))*(np.sin(theta)), (np.cos(fi))]) #direção do campo aplicado
    B = field*vec_u #Vetor campo magnético.
    for k in range(5000): #Esse for é para realizar a autoconsistência.
        Bef = campo_efetivo(M_0,g,lam,B) #calcula o campo efetivo de cada sub-rede.
        ener, psi = hamiltoniano(J,Bef,Hc) #calcula as autofunções e as energias de cada sub-rede.
        mag_subnets = magnetizacao(g,J,ener,psi,temp) #calcula a magnetização de cada sub-rede.
        delta = np.sum(np.abs(mag_subnets-M_0))
        if delta <= 0.0001: #autoconsistência.
            ener_livre = energia_livre(ener,temp)
            Entropia_subnets = entropia_magnetica(ener,temp) #entropia magnética de cada sub-rede
            #entropia_magnetica(ener,temp) pela constante universal dos gases perfeitos (R).
            mx, my, mz, Entropia, energia_total = 0, 0, 0, 0, 0
            for contador1 in range(len(P)):
                mx += P[contador1]*mag_subnets[contador1][0] #calcula a magnetização total na direção x
                my += P[contador1]*mag_subnets[contador1][1] #calcula a magnetização total na direção y
                mz += P[contador1]*mag_subnets[contador1][2] #calcula a magnetização total na direção z
                Entropia += (P[contador1]*Entropia_subnets[contador1]) #calcula a entropia total
            vetor_mag = (np.array([mx, my, mz])) #vetor magnetização resultante
            modulo_mag = (1/m_b)*(linalg.norm(vetor_mag))# modulo do vetor magnetização resultante
            if field == 0:
                MH = modulo_mag
            else:
                MH = (1/m_b)*(np.dot(vetor_mag,vec_u)) #Magnetização projetada na direção do campo magnético aplicado
            break
        else: # se a condição de delta não for satisfeita, ele faz o chute inicial ser o valor calculado em mag_subnets
            #realimentando assim o sistema
            M_0 = mag_subnets
    variaveis_saida = [modulo_mag, MH, mag_subnets/m_b, Entropia, Entropia_subnets,ener_livre]
    return variaveis_saida
'''
A função mag_campo(valores_entrada) calcula a projeção da magnetização valores_entrada = [J,Hc,g,lam,P,M_0,field,fi,theta,temp] 
é uma lista com esse conjunto de valores.A variável valores de entrada deve ser escrita igualmente para a função 
autoconsistencia(valores_entrada), com três excessões:
field deverá ser um array do tipo ([0,field1, field2, ..., fieldn]) fi = [fi1,fi2,..fin] e 
theta = [theta_1,theta_2,..theta_n], nesse caso, fi1 e theta_1 formam a 1º direção e assim sucessivamente. 
Mesmo que só tenha um valor, deve-se manter esse escrita.
'''
def mag_campo(valores_entrada):
    J,Hc,g = valores_entrada[0],valores_entrada[1],valores_entrada[2]
    lam,P,M_0 = valores_entrada[3], valores_entrada[4], valores_entrada[5]
    field,fi,theta,temp = valores_entrada[6], valores_entrada[7], valores_entrada[8], valores_entrada[9]
    Mh = np.zeros((len(field),len(fi)))
    for count2 in range(len(fi)):
        for count3 in range(len(field)):
            entrada = [J,Hc,g,lam,P,M_0,field[count3],fi[count2],theta[count2],temp] 
            saida_1 = autoconsistencia(entrada)
            Mh[count3,count2] = saida_1[1] 
    return Mh
'''
A função mag_temperatura(valores_entrada) calcula a projeção da magnetização total do sistema na direção do campo, 
a magnetização resultante do sistema, a magnetização resultante de cada sub-rede nas direções x, y, z, 
a entropia total, a entropia de cada sub-rede e a variação de entropia magnética do sistema.
A entrada identificada pela variável ''valores_entrada'' é uma lista 
com esse conjunto de valores ([J,Hc,g,lam,P,M_0,field,fi,theta,temp]).A variável valores de entrada deve ser 
escrita igualmente para a função autoconsistencia(valores_entrada), com duas excessões:
A primeira para field = [0.,field1,field2,..fieldn] e temp = [temp_1,temp_2,..temp_n]. Nesse caso, cada coluna 
da matriz de cada resultado dentro de valores_saida, terá o mesmo número de linhas que o array de temperatura 
e o mesmo número de colunas que o array field possui. A execessão será deltaS, que não terá o campo nulo.
Mesmo que só tenha um valor, deve-se manter esse escrita.
'''
def mag_temperatura(valores_entrada):
    J,Hc,g = valores_entrada[0],valores_entrada[1],valores_entrada[2]
    lam,P,M_0 = valores_entrada[3], valores_entrada[4], valores_entrada[5]
    field,fi,theta,temp = valores_entrada[6], valores_entrada[7], valores_entrada[8], valores_entrada[9]
    Mh, Mag, entropy = np.zeros((len(temp),len(field))),np.zeros((len(temp),len(field))),np.zeros((len(temp),len(field)))
    Mag_sub_x, Mag_sub_y, Mag_sub_z, entropy_sub = [],[],[],[]
    linhas, colunas = len(temp),len(field)
    for count1 in range(len(g)):
        Mag_sub_x.append(np.zeros((linhas, colunas)))
        Mag_sub_y.append(np.zeros((linhas, colunas)))
        Mag_sub_z.append(np.zeros((linhas, colunas)))
        entropy_sub.append(np.zeros((linhas, colunas)))
    for count2 in range(colunas):
        for count3 in range(linhas):
            entrada = [J,Hc,g,lam,P,M_0,field[count2],fi,theta,temp[count3]] 
            saida = autoconsistencia(entrada)
            m,MH,mag_sub = saida[0],saida[1],saida[2]
            s,s_sub = saida[3],saida[4]
            Mh[count3,count2] = MH
            Mag[count3,count2] = m
            entropy[count3,count2] = s
            for count4 in range(len(g)):
                Mag_sub_x[count4][count3,count2] = mag_sub[count4][0]
                Mag_sub_y[count4][count3,count2] = mag_sub[count4][1]
                Mag_sub_z[count4][count3,count2] = mag_sub[count4][2]
                entropy_sub[count4][count3,count2] = s_sub[count4]
    valores_saida = [Mh, Mag, Mag_sub_x, Mag_sub_y, Mag_sub_z, entropy, entropy_sub]
    return valores_saida

def delta_S_magnético(entropia_sem_campo, entropia_com_campo):
    eixos = entropia_com_campo.ndim
    if eixos==2:
        linhas = entropia_com_campo.shape[0]
        colunas = entropia_com_campo.shape[1]
        d_s = np.zeros((linhas,colunas))
        for count in range(colunas):
            for count0 in range(linhas):
                s0 = entropia_sem_campo[count0]
                s = entropia_com_campo[count0,count]
                d_s[count0,count] = s0 - s
    else:
        linhas = len(entropia_com_campo)
        d_s = np.zeros(linhas)
        for count in range(linhas):
            s0 = entropia_sem_campo[count]
            s = entropia_com_campo[count]
            d_s[count] = s0 - s
    return d_s
'''
A função free_energy_analysis(entrada) calcula a energia livre em função da temperatura para o campo magnético em uma 
direção fixa. A entrada deve ser do mesmo formato que a função mag_campo(valores_entrada). A saída será uma matriz 
cujo numero de linhas é igual ao range de temperatura (temp) e o número de colunas é igual ao range dos angulos fi e theta.
'''
def free_energy_analysis(entrada):
    J,Hc,g = entrada[0],entrada[1],entrada[2]
    lam,P,M_0 = entrada[3], entrada[4], entrada[5]
    field,fi,theta,temp = entrada[6], entrada[7], entrada[8], entrada[9]
    free_energy = np.zeros((len(temp),len(fi)))
    for j in range(len(fi)):
        for i in range(len(temp)):
            entrada = [J,Hc,g,lam,P,M_0,field,fi[j],theta[j],temp[i]]
            saida = autoconsistencia(entrada)
            free_energy[i,j] = saida[5]
    return free_energy

def calor_especifico(temperatura, entropia_magnetica, constantes,temperatura_deybe):
    gamma = constantes[0]
    N = constantes[1] #número de átomos na fórmula unitária
    R = 8.3144621 #Constante universal dos gases perfeitos
    funcao = lambda x: ((x**4)*(np.exp(x)))/(((np.exp(x))-1)**2) 
#do calor espeífico da rede
    dT = np.diff(temperatura)
    linhas = len(dT)
    eixos = entropia_magnetica.ndim
    if eixos == 2:
        colunas = entropia_magnetica.shape[1]
        Calor_especifico = np.zeros((linhas,colunas))
        for j in range(colunas):
            dS = np.diff(entropia_magnetica[:,j])
            cmag = dS/dT
            novo_temperatura = np.zeros(linhas)
            for i in range(linhas):
                novo_temperatura[i] = temperatura[i]
                xf = temperatura_deybe/temperatura[i] #Limite superior de integração 
                integral = integrate.quad(funcao, 0., xf)[0]#A função integrate.quad retorna 
#uma tupla sendo o primeiro elemento o valor da integral e o segundo o erro associado
                c_rede = N*R*(9.0*(xf)**(-3))*(integral)
                c_eletronico = gamma*temperatura[i]
                Calor_especifico[i,j] = temperatura[i]*cmag[i] + c_eletronico + c_rede
        return Calor_especifico, novo_temperatura
    elif eixos == 1:
        Calor_especifico = np.zeros(linhas)
        dS = np.diff(entropia_magnetica)
        cmag = dS/dT
        novo_temperatura = np.zeros(linhas)
        for i in range(linhas):
            novo_temperatura[i] = temperatura[i]
            xf = temperatura_deybe/temperatura[i] #Limite superior de integração 
            integral = integrate.quad(funcao, 0., xf)[0]#A função integrate.quad retorna 
#uma tupla sendo o primeiro elemento o valor da integral e o segundo o erro associado
            c_rede = N*R*(9.0*(xf)**(-3))*(integral)
            c_eletronico = gamma*temperatura[i]
            Calor_especifico[i] = temperatura[i]*cmag[i] + c_eletronico + c_rede
        return Calor_especifico, novo_temperatura
    else:
        print("ERRO ->>> O array entropia_magnetica deve ter 1 ou 2 eixos")
    
def entropia_total(temperatura, entropia_magnetica, constantes):
    gamma, temperatura_deybe = constantes[0],constantes[1]
    N = constantes[2] #número de átomos na fórmula unitária
    R = 8.3144621 #Constante universal dos gases perfeitos
    linhas = len(temperatura)
    funcao = lambda x: ((x**3)/(np.exp(x) - 1))
    eixos = entropia_magnetica.ndim
    if eixos == 2:
        colunas = entropia_magnetica.shape[1]
        entropia_total = np.zeros((linhas,colunas))
        for j in range(colunas):
            for i in range(linhas):
                fracao =  temperatura_deybe/temperatura[i]
                termo1 = (np.log(1 - np.exp((-1)*fracao)))
                integral = integrate.quad(funcao, 0., fracao)[0]
                termo2 = (-4)*((fracao)**(-3))*(integral)
                s_rede = (-3)*N*R*(termo1 + termo2)
                s_mag = entropia_magnetica[i,j]
                s_ele = gamma*temperatura[i]
                entropia_total[i,j] = s_rede + s_mag + s_ele
        return entropia_total
    elif eixos == 1:
        entropia_eletronica = gamma*temperatura
        entropia_da_rede = np.zeros(linhas)
        for i in range(linhas):
            fracao =  temperatura_deybe/temperatura[i]
            termo1 = (np.log(1 - np.exp((-1)*fracao)))
            integral = integrate.quad(funcao, 0., fracao)[0]
            termo2 = (-4)*((fracao)**(-3))*(integral)
            s_rede = (-3)*N*R*(termo1 + termo2)
            s_mag = entropia_magnetica[i]
            s_ele = gamma*temperatura[i]
            entropia_da_rede[i] = s_rede
        entropia_total = entropia_eletronica + entropia_da_rede + entropia_magnetica
        return entropia_total
    else:
        print("ERRO ->>> O array entropia_magnetica deve ter 1 ou 2 eixos")

def deltaT_adiabatico(temperatura, entropia_sem_campo, entropia_com_campo):
    inter_sem_campo = interpolate.interp1d(temperatura, entropia_sem_campo)
    L = len(temperatura)
    temp_min, delta_temp, temp_max = temperatura[0], 0.1, temperatura[L-1]
    temp_novo = np.arange(temp_min, temp_max, delta_temp)
    entro_sem_campo_novo = inter_sem_campo(temp_novo)
    eixos = entropia_com_campo.ndim
    if eixos == 2:
        DT_adiabatico, T = [], []
        colunas = entropia_com_campo.shape[1]
        for contador in range(colunas):
            S = entropia_com_campo[:,contador]
            inter_com_campo = interpolate.interp1d(temperatura, S)
            entro_com_campo_novo = inter_com_campo(temp_novo)
            contador1 = len(temp_novo)
            temp,deltat_ad = [],[]
            for i in range(contador1):
                t, s0 = temp_novo[i],entro_sem_campo_novo[i]
                for j in range(contador1-1):
                    th1, th2 = temp_novo[j], temp_novo[j+1]
                    sh1, sh2 = entro_com_campo_novo[j], entro_com_campo_novo[j + 1]
                    s1, s2 = s0 - sh1, s0 - sh2
                    f = s1*s2
                    if s1 == 0:
                        deltaT = th1 - t
                        deltat_ad.append(deltaT)
                        temp.append(t)
                        break
                    elif s2 == 0:
                        deltaT = th2 - t
                        deltat_ad.append(deltaT)
                        temp.append(t)
                        break
                    elif f < 0:
                        for k in range(51):
                            th3 = (th1 + th2)/2
                            sh3 = inter_com_campo(th3)
                            s3 = s0 - sh3
                            f1 = s1*s3
                            if  s3 == 0:
                                deltaT = th3 - t
                                deltat_ad.append(deltaT)
                                temp.append(t)
                                break
                            elif f1 > 0:
                                th1 = th3
                                sh1 = sh3
                            else:
                                th2 = th3
                                sh2 = sh3
                            if k == 50:
                                deltaT = th3 - t
                                deltat_ad.append(deltaT)
                                temp.append(t)
                    else:
                        continue
            x, y = np.array(temp), np.array(deltat_ad)
            DT_adiabatico.append(y)
            T.append(x)
        return T, DT_adiabatico
    elif eixos == 1:
        inter_com_campo = interpolate.interp1d(temperatura, entropia_com_campo)
        entro_com_campo_novo = inter_com_campo(temp_novo)
        contador1 = len(temp_novo)
        temp,deltat_ad = [],[]
        for i in range(contador1):
            t, s0 = temp_novo[i],entro_sem_campo_novo[i]
            for j in range(contador1-1):
                th1, th2 = temp_novo[j], temp_novo[j+1]
                sh1, sh2 = entro_com_campo_novo[j], entro_com_campo_novo[j + 1]
                s1, s2 = s0 - sh1, s0 - sh2
                f = s1*s2
                if s1 == 0:
                    deltaT = th1 - t
                    deltat_ad.append(deltaT)
                    temp.append(t)
                    break
                elif s2 == 0:
                    deltaT = th2 - t
                    deltat_ad.append(deltaT)
                    temp.append(t)
                    break
                elif f < 0:
                    for k in range(51):
                        th3 = (th1 + th2)/2
                        sh3 = inter_com_campo(th3)
                        s3 = s0 - sh3
                        f1 = s1*s3
                        if  s3 == 0:
                            deltaT = th3 - t
                            deltat_ad.append(deltaT)
                            temp.append(t)
                            break
                        elif f1 > 0:
                            th1 = th3
                            sh1 = sh3
                        else:
                            th2 = th3
                            sh2 = sh3
                        if k == 50:
                            deltaT = th3 - t
                            deltat_ad.append(deltaT)
                            temp.append(t)
                else:
                    continue
            T, DT_adiabatico = np.array(temp), np.array(deltat_ad)
        return T, DT_adiabatico
    else:
        print("ERRO ->>> O array entropia_magnetica deve ter 1 ou 2 eixos")
