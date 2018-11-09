import cifar10_wgangp
import mnist_cwgangp
import mnist_gan
import mnist_wgangp

if __name__ == '__main__':
    mnist_gan.main(['--save-directory=output/mnist_gan/freq5'])
    mnist_gan.main(['--save-directory=output/mnist_gan/freq1', '--generator-frequency=1'])
    mnist_wgangp.main(['--save-directory=output/mnist_wgangp'])
    mnist_cwgangp.main(['--save-directory=output/mnist_cwgangp'])
    cifar10_wgangp.main(['--save-directory=output/cifar10_wgangp'])
