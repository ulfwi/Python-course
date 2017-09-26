"""
Exact and inexact line search methods (Wolfe, Goldstein)

Authors: Fanny Andersson (tna13fan), Louise Sjöholm (tfy13lsj), Lina Sjöstrand (fra12lsj), Björn Ulfwi (tfy13bul)
"""

import scipy.optimize as op
import numpy as np


def func_alpha(func, alp, x, p):
    """
    Define objective function as a function of alpha

    :param func: objective function
    :param alp: step length
    :param p: search direction
    :param x: point
    :return: obj function evaluated in point (x + alpha*p)
    """
    return func(x + alp * p)


def grad_alpha(grad, alp, x, p):
    """
    Define derivative of objective function with respect to alpha

    :param grad: gradient of objective function
    :param alp: step length
    :param p: search direction
    :param x: point
    :return: derivative of obj function evaluated in point (x + alpha*p)
    """
    return np.dot(grad(x + alp * p), p)


def ls_exact(func, x, p):
    """
    Exact line search. Minimize objective function func with respect to alpha (alp)
    using scipy.optimize.fmin
    
    :param func: function to be minimized
    :param p: direction of steepest descent
    :param x: evaluation point
    :return: step length alpha (alp)
    """
    def f(alpha, x, p):
        return func(x+alpha*p)
    alp = 0
    alp = op.fmin(f, alp, args=(x, p), disp=False)
    if alp < 0:
        raise ArithmeticError("alpha cannot be negative!")
    return alp


def ls_block1(grad, x, p, alp_0, alp_l, tau=0.1, chi=9):
    """
    Block 1 in algorithm for inexact line search (see p. 48 in lecture notes)

    :param grad: gradient of objective function
    :param x: current point
    :param p: search direction
    :param alp_0: soon to be acceptable point (when rc and lc are true)
    :param alp_l: lower bound of alpha interval
    :param tau: parameter in algorithm
    :param chi: parameter in algorithm
    :return: updated alp_0 and alp_l
    """
    # Extrapolate
    d_alp_0 = (alp_0 - alp_l) * (grad_alpha(grad, alp_0, x, p)) / \
              (grad_alpha(grad, alp_l, x, p) - grad_alpha(grad, alp_0, x, p))
    d_alp_0 = max([d_alp_0, tau * (alp_0 - alp_l)])
    d_alp_0 = min([d_alp_0, chi * (alp_0 - alp_l)])
    alp_l = alp_0
    alp_0 = alp_0 + d_alp_0

    return alp_0, alp_l


def ls_block2(func, grad, x, p, alp_0, alp_l, alp_u, tau=0.1):
    """
     Block 2 in algorithm for inexact line search (see p. 48 in lecture notes)

    :param func: objective function
    :param grad: gradient of objective function
    :param x: current point
    :param p: direction of steepest descent
    :param alp_0: soon to be acceptable point (when rc and lc are true)
    :param alp_u: upper bound of alpha interval
    :param alp_l: lower bound of alpha interval
    :param tau: parameter in algorithm
    :return: updated alp_0 and alp_u
    """
    alp_u = min([alp_0, alp_u])
    # Interpolate
    alp_0_bar = ((alp_0 - alp_l) ** 2 * grad_alpha(grad, alp_l, x, p)) / \
                (2 * (func_alpha(func, alp_l, x, p) - func_alpha(func, alp_0, x, p) +
                      (alp_0 - alp_l)*grad_alpha(grad, alp_l, x, p)))
    alp_0_bar = max([alp_0_bar, alp_l + tau * (alp_u - alp_l)])
    alp_0_bar = min([alp_0_bar, alp_u - tau * (alp_u - alp_l)])
    alp_0 = alp_0_bar

    return alp_0, alp_u


def ls_gold(func, grad, x, p, alp_0=1, rho=0.25):
    """
    Line search with Goldstein conditions

    :param func: objective function
    :param grad: gradient of objective function
    :param x: point
    :param p: direction of steepest descent
    :param alp_0: soon to be acceptable point (when rc and lc are true)
    :param rho: parameter in algorithm
    :return: step length alp_0, acceptable point having rc and lc true
    """
    # Initialize upper and lower bound of alpha (alp)
    alp_l = 0
    alp_u = 10 ** 99

    # Check if rho is between [0, 0.5]
    if rho > 0.5 or rho < 0:
        rho = 0.1
        print('Your choice of rho was out of bounds (allowed bounds [0, 0.5]), it has now been set to 0.1')

    # Check if lc (left condition) and rc (right condition) is true or false
    lc = (func_alpha(func, alp_0, x, p) >= func_alpha(func, alp_l, x, p) + (1. - rho) * (alp_0 - alp_l)
          * grad_alpha(grad, alp_l, x, p))
    rc = (func_alpha(func, alp_0, x, p) <= func_alpha(func, alp_l, x, p) + rho * (alp_0 - alp_l)
          * grad_alpha(grad, alp_l, x, p))

    # While lc or rc is false, update alp_l and alp_0 or alp_u and alp_0 respectively
    while ~(lc & rc):
        if ~lc:
            alp_0, alp_l = ls_block1(grad, x, p, alp_0, alp_l)
        # if ~rc:
        else:
            alp_0, alp_u = ls_block2(func, grad, x, p, alp_0, alp_l, alp_u)
        lc = (func_alpha(func, alp_0, x, p) >= func_alpha(func, alp_l, x, p) +
              (1. - rho) * (alp_0 - alp_l) * grad_alpha(grad, alp_l, x, p))
        rc = (func_alpha(func, alp_0, x, p) <= func_alpha(func, alp_l, x, p) +
              rho * (alp_0 - alp_l) * grad_alpha(grad, alp_l, x, p))

    return alp_0


def ls_wolfe(func, grad, x, p, alp_0=1, rho=0.1, sigma=0.7):
    """
    Line search with Wolfe-Powell conditions

    :param func: objective function
    :param grad: gradient of objective function
    :param x: current point
    :param p: direction of steepest descent
    :param alp_0: soon to be acceptable point (when rc and lc are true)
    :param rho: parameter in algorithm
    :param sigma: parameter in algorithm
    :return: step length alp_0, acceptable point having rc and lc true
    """
    # Initialize upper and lower bound of alpha (alp)
    alp_l = 0
    alp_u = 10 ** 99

    # Check if rho is between [0, 0.5]
    if rho > 0.5 or rho < 0:
        rho = 0.1
        print('Your choice of rho was out of bounds (allowed bounds [0, 0.5]), it has now been set to 0.1')

    # Check if sigma is bigger than rho
    if sigma > 1 or sigma < 0:
        sigma = 0.7
        print('Your choice of sigma was out of bounds (allowed bounds [0, 1], it is now set to 0.7')
    if sigma <= rho:
        sigma = 0.7
        print('Your choice of sigma was inaccurate since sigma > rho, it has now been set to 0.7')

    # Check if lc (left condition) and rc (right condition) is true or false
    lc = (grad_alpha(grad, alp_0, x, p) >= sigma * grad_alpha(grad, alp_l, x, p))
    rc = (func_alpha(func, alp_0, x, p) <= func_alpha(func, alp_l, x, p) + rho * (alp_0 - alp_l)
          * grad_alpha(grad, alp_l, x, p))

    # While lc or rc is false, update alp_l and alp_0 or alp_u and alp_0 respectively
    while ~(lc & rc):
        if ~lc:
            alp_0, alp_l = ls_block1(grad, x, p, alp_0, alp_l)
        # if ~rc:
        else:
            alp_0, alp_u = ls_block2(func, grad, x, p, alp_0, alp_l, alp_u)
        lc = (grad_alpha(grad, alp_0, x, p) >= sigma * grad_alpha(grad, alp_l, x, p))
        rc = (func_alpha(func, alp_0, x, p) <= func_alpha(func, alp_l, x, p)
              + rho * (alp_0 - alp_l) * grad_alpha(grad, alp_l, x, p))

    return alp_0

