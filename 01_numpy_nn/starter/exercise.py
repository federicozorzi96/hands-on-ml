# Jupytext-compatible percent format; converted from rete_neurale_numpy_guidata.ipynb

# %% [markdown]
# # Una piccola rete neurale da zero con NumPy
#
# Costruiamo una rete **2 → 4 → 1** per imparare XOR. Useremo solo NumPy per i calcoli: niente framework, niente autodiff.
#
# ## Come usarlo
# 1. Leggi la traccia guidata e completa le parti `TODO` senza scendere alla soluzione.
# 2. Esegui le celle una alla volta e osserva le forme dei vettori e la loss.
# 3. Solo dopo confronta il tuo lavoro con la soluzione completa in fondo.
#
# La rete calcola: `X → Z1 → A1 → Z2 → ŷ`. Durante il training, la backpropagation applica la regola della catena per ottenere i gradienti dei pesi.

# %% [markdown]
# ## 1. Dati XOR
#
# XOR vale 1 quando esattamente uno dei due ingressi è 1. Una singola retta non può separare queste quattro osservazioni: serve almeno uno strato nascosto.

# %%
import numpy as np

np.random.seed(7)
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])
print(X.shape, y.shape)  # (4, 2), (4, 1)

# %% [markdown]
# ## 2. Traccia guidata: completa i TODO
#
# ### Sigmoide e inizializzazione
# Usiamo sigmoide per trasformare i valori in probabilità. Inizializza matrici di pesi con dimensioni coerenti: `W1` collega 2 ingressi a 4 neuroni; `W2` collega 4 neuroni all'uscita.
#
# **Domande:** perché aggiungiamo un bias? Quali devono essere le forme di `b1` e `b2`?

# %%
def sigmoid(z):
    # TODO: implementa 1 / (1 + exp(-z))
    raise NotImplementedError

n_in, n_hidden, n_out = 2, 4, 1
W1 = ...  # TODO: inizializza una matrice (2, 4), valori piccoli casuali
b1 = ...  # TODO: zeri, forma (1, 4)
W2 = ...  # TODO: inizializza una matrice (4, 1)
b2 = ...  # TODO: zeri, forma (1, 1)

# %% [markdown]
# ### Forward pass e loss
#
# Per ogni strato: `Z = A @ W + b`, poi applica la sigmoide. Calcola la binary cross-entropy media. Aggiungere un piccolo epsilon evita `log(0)`.
#
# **Controllo:** `y_hat` deve avere forma `(4, 1)`; la loss deve essere uno scalare.

# %%
def forward(X, W1, b1, W2, b2):
    # TODO: calcola Z1, A1, Z2, y_hat
    # Restituisci anche Z1 e A1: serviranno nella backpropagation
    raise NotImplementedError

def binary_cross_entropy(y, y_hat):
    # TODO: media di -(y*log(y_hat) + (1-y)*log(1-y_hat))
    raise NotImplementedError

# %% [markdown]
# ### Backpropagation
#
# Per BCE con sigmoide in uscita, il gradiente rispetto a `Z2` si semplifica in `y_hat - y`. Poi usa la regola della catena per risalire allo strato nascosto. Ricorda che la derivata della sigmoide è `A * (1 - A)`.
#
# **Domanda:** perché per i pesi facciamo un prodotto tra attivazioni trasposte e gradiente dello strato successivo?

# %%
def gradients(X, y, Z1, A1, y_hat, W2):
    m = X.shape[0]
    # TODO: calcola dZ2, dW2, db2, dZ1, dW1, db1
    # Dividi i gradienti per m
    raise NotImplementedError

# %% [markdown]
# ### Aggiornamento dei parametri
#
# La discesa del gradiente sposta ogni parametro nella direzione opposta al gradiente: `parametro -= learning_rate * gradiente`. Esegui molti passaggi sullo stesso piccolo dataset XOR.

# %%
learning_rate = 1.0
epochs = 10_000

for epoch in range(epochs):
    # TODO: forward, loss, gradients, aggiornamento di W1, b1, W2, b2
    pass

# Stampa loss e probabilità predette
