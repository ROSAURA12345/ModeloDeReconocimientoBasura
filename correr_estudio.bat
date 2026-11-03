@echo off
REM Uso:  correr_estudio.bat gpu     (o cpu)
REM Se puede cortar y volver a lanzar: la etapa 4 retoma donde quedó.
set EQ=%1
if "%EQ%"=="" set EQ=auto
call venv\Scripts\activate
python etapas\e2_inventario.py || goto :error
if not exist datos\derivados\pliegues.csv python etapas\e3_pliegues.py || goto :error
python etapas\e4_estudio_factorial.py --equipo %EQ% || goto :error
python etapas\e5_analisis_estadistico.py || goto :error
python etapas\e6_modelo_final.py --equipo %EQ% || goto :error
python etapas\e8_mapas_gradcampp.py --equipo %EQ%
echo.
echo Informes: resultados\estudio\informe_estadistico.md  y  resultados\final\informe_prueba.md
pause
exit /b 0
:error
echo Hubo un error. Lea el mensaje de arriba.
pause
exit /b 1
