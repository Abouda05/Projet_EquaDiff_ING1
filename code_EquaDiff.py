import numpy as np
import os
from scipy import integrate, optimize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def construire_P(g):
    def P(x):
        if x <= 0:
            return np.nan
        valeur, _ = integrate.quad(g, 0.0, x, limit=200)
        return -2.0 * valeur / x**2
    return P


def construire_G(g):
    def G(y):
        valeur, _ = integrate.quad(g, 0.0, y, limit=200)
        return valeur
    return G


def derivee_numerique_P(P, x, pas=1e-5):
    return (P(x + pas) - P(x - pas)) / (2.0 * pas)


def trouver_parametres(g, x_max=10.0, nb_points=2000):
    P = construire_P(g)

    xs = np.linspace(1e-3, x_max, nb_points)
    valeurs_dP = np.array([derivee_numerique_P(P, x) for x in xs])

    changements_signe = np.where(np.diff(np.sign(valeurs_dP)))[0]

    candidats = []
    for idx in changements_signe:
        try:
            c_etoile = optimize.brentq(
                lambda x: derivee_numerique_P(P, x),
                xs[idx],
                xs[idx + 1],
                xtol=1e-10,
                rtol=1e-10,
            )
        except Exception:
            continue

        mu_etoile = P(c_etoile)
        if mu_etoile <= 0.0:
            continue

        x_verification = np.linspace(1e-4, c_etoile - 1e-4, 500)
        P_verification = np.array([P(x) for x in x_verification])
        if np.all(P_verification < mu_etoile - 1e-10):
            candidats.append((float(c_etoile), float(mu_etoile)))

    return candidats


def second_membre(x, etat, mu, G):
    Q = np.clip(etat[0], 1e-12, None)
    expression_sous_racine = max(mu * Q**2 + 2.0 * G(Q), 0.0)
    return [-np.sqrt(expression_sous_racine)]


def resoudre_et_tracer(g, c, mu, intervalle=(-20.0, 20.0), nb_points=2000, ax=None):
    G = construire_G(g)
    Q_initial = c / 2.0

    solution = integrate.solve_ivp(
        fun=lambda x, etat: second_membre(x, etat, mu, G),
        t_span=intervalle,
        y0=[Q_initial],
        method='RK45',
        t_eval=np.linspace(intervalle[0], intervalle[1], nb_points),
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10,
    )

    xs = solution.t
    Qs = solution.y[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(xs, Qs, color='steelblue', lw=2)
    ax.axhline(c, color='tomato', ls='--', lw=1.2, label='c = {:.4f}'.format(c))
    ax.axhline(0, color='gray', ls='--', lw=1.2, label='0')
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('Q(x)', fontsize=13)
    ax.set_title('Solution pour c = {:.4f}, mu = {:.4f}'.format(c, mu), fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05 * c, 1.15 * c)

    return xs, Qs


def visualiser_tout(g, nom_g="g", x_max=8.0, dossier_sortie="outputs"):
    print("=== Analyse pour g = {} ===\n".format(nom_g))
    P = construire_P(g)
    parametres = trouver_parametres(g, x_max=x_max)

    if not parametres:
        print("Aucun couple (c, mu) trouvé dans [0, {}].".format(x_max))
        print("Essayez d'augmenter x_max ou vérifiez que g satisfait g(0) = g'(0) = 0.")
        return

    print("Nombre de couples (c, mu) trouvés : {}".format(len(parametres)))
    for i, (c, mu) in enumerate(parametres):
        print("  [{}]  c = {:.6f},  mu = {:.6f}".format(i + 1, c, mu))

    nb = len(parametres)
    fig = plt.figure(figsize=(14, 4 * (nb + 1)))
    grille = GridSpec(nb + 1, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_P = fig.add_subplot(grille[0, :])
    xs_trace = np.linspace(1e-2, x_max, 800)
    Ps_trace = np.array([P(x) for x in xs_trace])
    ax_P.plot(xs_trace, Ps_trace, 'k', lw=2, label='P(x)')

    for c, mu in parametres:
        ax_P.axvline(c, color='tomato', ls='--', lw=1.2)
        ax_P.plot(c, mu, 'ro', ms=8, zorder=5,
                  label='(c, mu) = ({:.3f}, {:.3f})'.format(c, mu))

    ax_P.axhline(0, color='gray', lw=0.8)
    ax_P.set_xlabel('x', fontsize=12)
    ax_P.set_ylabel('P(x)', fontsize=12)
    ax_P.set_title('P(x) = -2G(x)/x²  pour  g = {}'.format(nom_g), fontsize=13)
    ax_P.legend(fontsize=10)
    ax_P.grid(True, alpha=0.3)

    for i, (c, mu) in enumerate(parametres):
        ax_Q = fig.add_subplot(grille[i + 1, :])
        resoudre_et_tracer(g, c, mu, intervalle=(-15.0, 15.0), ax=ax_Q)

    fig.suptitle(
        "-Q'' + mu*Q + g(Q) = 0   pour   g = {}".format(nom_g),
        fontsize=14,
        y=1.01,
    )

    os.makedirs(dossier_sortie, exist_ok=True)
    nom_fichier = "{}/solution_{}.png".format(dossier_sortie, nom_g)
    plt.savefig(nom_fichier, dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure enregistrée : {}\n".format(nom_fichier))


if __name__ == "__main__":

    def g1(t):
        return t**3 - t**2
    visualiser_tout(g1, nom_g="t3_moins_t2", x_max=6.0)

    def g2(t):
        return t**2 * (t - 2.0)
    visualiser_tout(g2, nom_g="t2_fois_t_moins_2", x_max=8.0)

    def g3(t):
        return t**2 * np.sin(t)
    visualiser_tout(g3, nom_g="t2_sin_t", x_max=8.0)

    def g4(t):
        return t**3 - 2.0 * t**2
    visualiser_tout(g4, nom_g="t3_moins_2t2", x_max=8.0)

    print("Terminé")