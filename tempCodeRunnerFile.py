# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))