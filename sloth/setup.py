from distutils.core import setup

setup(name='Sloth',
    version='2.0.6',
    description='Time series tools for classification, forecasting and clustering',
    packages=['Sloth'],
    install_requires=['scikit-learn >= 0.18.1',
        'fastdtw>=0.3.2',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'numpy>=1.14.2',
        'matplotlib>=2.2.2',
        'statsmodels>=0.9.0',
        'pyramid-arima>=0.6.5',
        'cython>=0.28.5',
        'tslearn @ git+https://github.com/NewKnowledge/tslearn@6eb333fa1606d90fbdb37f975f3ffbe265b91198#egg=tslearn-0.1.28.3',
        'hdbscan>=0.8.18', 
        'Keras>=2.1.6',
        'tensorflow-gpu <= 1.12.0',
        'seaborn'],
    include_package_data=True,
)