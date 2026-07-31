"""Suite de tests del MCMC.

Los tests test_regression_* verifican consistencia interna y estabilidad
de los valores calibrados (que el código sigue produciendo lo que declara).
NO constituyen contraste observacional: eso requiere los datos de data/
y el ajuste bayesiano de producción.

El test del límite de recuperación (test_recovery_limit.py) verifica la
Proposición A.1 del Tratado de Fundamentos: con ε → 0 el modelo devuelve
exactamente ΛCDM.
"""
