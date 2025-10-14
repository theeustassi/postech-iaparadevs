@echo off
setlocal

:: Verifica se um nome foi fornecido
if "%~1"=="" (
    echo Uso: %~n0 nome_da_estrutura
    exit /b 1
)

:: Nome da estrutura
set "STRUCTURE_NAME=%~1"

:: Cria a estrutura de diretórios
mkdir "%STRUCTURE_NAME%\%STRUCTURE_NAME%"
mkdir "%STRUCTURE_NAME%\tests"

:: Cria o __init__.py
type nul > "%STRUCTURE_NAME%\%STRUCTURE_NAME%\__init__.py"

:: Adiciona o código da biblioteca no core.py
if not exist "%STRUCTURE_NAME%\%STRUCTURE_NAME%" mkdir "%STRUCTURE_NAME%\%STRUCTURE_NAME%"
echo # %STRUCTURE_NAME%/core.py > "%STRUCTURE_NAME%\%STRUCTURE_NAME%\core.py"
echo def hello_world()^: >> "%STRUCTURE_NAME%\%STRUCTURE_NAME%\core.py"
echo     return "Hello, world!" >> "%STRUCTURE_NAME%\%STRUCTURE_NAME%\core.py"

:: Adiciona um teste básico no test_core.py
if not exist "%STRUCTURE_NAME%\tests" mkdir "%STRUCTURE_NAME%\tests"
echo import pytest > "%STRUCTURE_NAME%\tests\test_core.py"
echo import sys >> "%STRUCTURE_NAME%\tests\test_core.py"
echo import os >> "%STRUCTURE_NAME%\tests\test_core.py"
echo sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) >> "%STRUCTURE_NAME%\tests\test_core.py"
echo from %STRUCTURE_NAME%.core import hello_world >> "%STRUCTURE_NAME%\tests\test_core.py"
echo def test_hello_world()^: >> "%STRUCTURE_NAME%\tests\test_core.py"
echo     assert hello_world() == "Hello, world!" >> "%STRUCTURE_NAME%\tests\test_core.py"

:: Cria o setup.py com a configuração do pacote
echo from setuptools import setup, find_packages > "%STRUCTURE_NAME%\setup.py"
echo with open("README.md", "r", encoding="utf-8") as fh: >> "%STRUCTURE_NAME%\setup.py"
echo     long_description = fh.read() >> "%STRUCTURE_NAME%\setup.py"
echo setup( >> "%STRUCTURE_NAME%\setup.py"
echo     name='%STRUCTURE_NAME%-package', >> "%STRUCTURE_NAME%\setup.py"
echo     version='1.0.0', >> "%STRUCTURE_NAME%\setup.py"
echo     packages=find_packages(), >> "%STRUCTURE_NAME%\setup.py"
echo     description='Descricao da sua lib %STRUCTURE_NAME%', >> "%STRUCTURE_NAME%\setup.py"
echo     author='seu nome', >> "%STRUCTURE_NAME%\setup.py"
echo     author_email='seu.email@example.com', >> "%STRUCTURE_NAME%\setup.py"
echo     url='https://github.com/tadrianonet/%STRUCTURE_NAME%', >> "%STRUCTURE_NAME%\setup.py"
echo     license='MIT', >> "%STRUCTURE_NAME%\setup.py"
echo     long_description=long_description, >> "%STRUCTURE_NAME%\setup.py"
echo     long_description_content_type='text/markdown' >> "%STRUCTURE_NAME%\setup.py"
echo ^) >> "%STRUCTURE_NAME%\setup.py"

:: Cria o README.md com instruções básicas
echo # %STRUCTURE_NAME% > "%STRUCTURE_NAME%\README.md"
echo A simple example library. >> "%STRUCTURE_NAME%\README.md"
echo ## Installation >> "%STRUCTURE_NAME%\README.md"
echo ```sh >> "%STRUCTURE_NAME%\README.md"
echo pip install %STRUCTURE_NAME% >> "%STRUCTURE_NAME%\README.md"
echo ``` >> "%STRUCTURE_NAME%\README.md"
echo ## Usage >> "%STRUCTURE_NAME%\README.md"
echo ```python >> "%STRUCTURE_NAME%\README.md"
echo from %STRUCTURE_NAME% import hello_world >> "%STRUCTURE_NAME%\README.md"
echo print(hello_world()) >> "%STRUCTURE_NAME%\README.md"
echo ``` >> "%STRUCTURE_NAME%\README.md"

echo Estrutura %STRUCTURE_NAME% criada com sucesso!

endlocal