import os
import sys

MKLDNN_ROOT = os.environ['HOME'] + '/.chainer'
MKLDNN_WORK_PATH = os.path.split(os.path.realpath(__file__))[0]
MKLDNN_LIB_PATH = MKLDNN_ROOT + '/lib'
MKLDNN_INCLUDE_PATH = MKLDNN_ROOT + '/include'
MKLDNN_SOURCE_PATH = MKLDNN_WORK_PATH + '/source'
MKLML_PKG_PATH = MKLDNN_SOURCE_PATH + '/external'


def download(mkldnn_version):
    print('Downloading ...')

    os.chdir(MKLDNN_WORK_PATH)
    os.system('git clone -b master --single-branch https://github.com/01org/mkl-dnn.git source')

    os.chdir(MKLDNN_SOURCE_PATH)
    os.system('git reset --hard %s' % mkldnn_version)


def install():
    print('Installing ...')

    os.chdir(MKLDNN_SOURCE_PATH)

    # install mkldnn
    if not os.path.exists(MKLML_PKG_PATH):
        os.system('cd scripts && ./prepare_mkl.sh && cd ..')
    os.system('mkdir -p build && cd build && cmake -DCMAKE_INSTALL_PREFIX=%s .. && make' % MKLDNN_ROOT)
    os.system('cd build && make install')

    # install mklml
    mklml_pkg_path_leafs = os.listdir(MKLML_PKG_PATH)
    mklml_origin_path = None
    for leaf in mklml_pkg_path_leafs:
        if os.path.isdir('%s/%s' % (MKLML_PKG_PATH, leaf)) and \
           'mklml' in leaf:
            mklml_origin_path = '%s/%s' % (MKLML_PKG_PATH, leaf)
            break

    if mklml_origin_path:
        os.system('cp %s/lib/* %s' % (mklml_origin_path, MKLDNN_LIB_PATH))
        os.system('cp %s/include/* %s' % (mklml_origin_path, MKLDNN_INCLUDE_PATH))


def download_and_install(mkldnn_version):
    download(mkldnn_version)
    install()


def prepare(mkldnn_version):
    print('Intel mkl-dnn preparing ...')
    mkldnn_prepared = True
    mkldnn_installed = True

    if os.path.exists(MKLDNN_SOURCE_PATH):
        os.chdir(MKLDNN_SOURCE_PATH)
        res = os.popen('git log | sed -n \'1p\'', 'r')
        commit_head = res.read()
        if mkldnn_version not in commit_head:
            os.system('rm -rf *')
            mkldnn_prepared = False
        else:
            if not os.path.exists(MKLDNN_LIB_PATH) or \
               not os.path.exists(MKLDNN_INCLUDE_PATH):
                os.system('rm -rf %s %s' % (MKLDNN_LIB_PATH, MKLDNN_INCLUDE_PATH))
                mkldnn_installed = False
    else:
        mkldnn_prepared = False

    if not mkldnn_prepared:
        download_and_install(mkldnn_version)
    elif not mkldnn_installed:
        install()

    os.chdir(sys.path[0])
    print('Intel mkl-dnn prepared !')


def root():
    return MKLDNN_ROOT


def lib_path():
    return MKLDNN_LIB_PATH


def include_path():
    return MKLDNN_INCLUDE_PATH


def source_path():
    return MKLDNN_SOURCE_PATH
