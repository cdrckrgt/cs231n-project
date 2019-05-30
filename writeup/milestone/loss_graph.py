from matplotlib import pyplot as plt

d_loss = [1, 2, 3, 4, 5]
g_loss = [5, 4, 3, 2, 1]
plt.plot(d_loss, label='Discriminator Loss')
plt.plot(g_loss, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
