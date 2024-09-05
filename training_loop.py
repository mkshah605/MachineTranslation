import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch as t

from IPython.display import display
from torch import optim
from torch import nn
from tqdm import tqdm

device = t.device("mps")



class TrainingLoop:
    def __init__(self):
        """
        """

    # def asMinutes(self, s):
    #     m = math.floor(s / 60)
    #     s -= m * 60
    #     return '%dm %ds' % (m, s)


    # def timeSince(self, since, percent):
    #     now = time.time()
    #     s = now - since
    #     es = s / (percent)
    #     rs = es - s
    #     return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))


    # def showPlot(self, points):
    #     plt.figure()
    #     fig, ax = plt.subplots()
    #     # this locator puts ticks at regular intervals
    #     loc = ticker.MultipleLocator(base=0.2)
    #     ax.yaxis.set_major_locator(loc)
    #     plt.plot(points)


    # def train_epoch(self, dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    #     total_loss = 0
    #     for batch in tqdm(dataloader):
    #         input_tensor, target_tensor = batch

    #         encoder_optimizer.zero_grad()
    #         decoder_optimizer.zero_grad()

    #         encoder_output, encoder_output_hidden = encoder(input_tensor)
    #         encoder_out = t.transpose(encoder_output_hidden[1], 0, 1)
    #         output, hidden = decoder(encoder_out)

    #         loss = criterion(
    #             output.view(-1, output.size(-1)),
    #             target_tensor.view(-1)
    #         )
    #         loss.backward()

    #         print(loss)

    #         encoder_optimizer.step()
    #         decoder_optimizer.step()

    #         total_loss += loss.item()

    #     return total_loss / len(dataloader)



    # NEW ADDED
    def training_step(self, batch, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, device="mps"):
        '''
        '''
        batch = [tensor.to(device) for tensor in batch]
        input_tensor, target_tensor = batch
        encoder_output, encoder_output_hidden = encoder(input_tensor)
        encoder_out = t.transpose(encoder_output_hidden[1], 0, 1).to(device)
        output, hidden = decoder(encoder_out)
        loss = criterion(
            output.view(-1, output.size(-1)), target_tensor.view(-1)
        )
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss
    
    # to be used with Attention layer?
    def training_step2(self, batch, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion):
        '''
        '''
        input_tensor, target_tensor = batch
        encoder_output, encoder_output_hidden = encoder(input_tensor)
        encoder_out_hidden = t.transpose(encoder_output_hidden[1], 0, 1)
        output, hidden = decoder(encoder_output, encoder_out_hidden)
        loss = criterion(
            output.view(-1, output.size(-1)), target_tensor.view(-1)
        )
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss

    

    def validation_step(self, batch, encoder, decoder):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for
        the whole validation set).
        '''
        input_tensor, target_tensor = batch
        encoder_output, encoder_output_hidden = encoder(input_tensor)
        encoder_out = t.transpose(encoder_output_hidden[1], 0, 1)
        output, hidden = decoder(encoder_out)

        predictions = output.argmax(dim=-1)
        correct = (predictions == target_tensor).flatten()
        return correct



    def train(self, train_dataloader, val_dataloader, encoder, decoder, n_epochs, learning_rate=0.001):
        # #print_every=100, plot_every=100):
        
        # # my old code
        # start = time.time()
        # plot_losses = []
        # print_loss_total = 0  # Reset every print_every
        # plot_loss_total = 0  # Reset every plot_every

        # Fern's plot code
        accuracy = np.nan

        progress_bar = tqdm(total = len(train_dataloader) * n_epochs)

        loss_logs = []
        loss_fig = plt.figure()
        loss_ax = plt.gca()
        # loss_display = display(loss_fig, display_id=True)
        
        acc_logs = []
        acc_fig = plt.figure()
        acc_ax = plt.gca()
        # acc_display = display(acc_fig, display_id=True) 

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.NLLLoss() #negative log likelihood loss

        for epoch in range(0, n_epochs):
            step = 0
            for i, batch in enumerate(train_dataloader):
                step += 1
                loss = self.training_step(batch, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}, loss: {loss:.3f}, accuracy: {accuracy:.2f}")

                loss_logs.append(np.array(object=loss.detach().to("cpu")))
            
                if step % 100 == 0:
                    loss_ax.clear()
                    loss_ax.plot(loss_logs[100:])
                    loss_fig.canvas.draw()
                    # loss_display.update(obj=loss_fig)

            correct_predictions = t.concat([self.validation_step(batch, encoder, decoder) for batch in val_dataloader])
            accuracy = correct_predictions.detach().to(device).float().mean().item()

            acc_logs.append(np.array(object=accuracy))
            acc_ax.clear()
            acc_ax.plot(acc_logs)
            acc_fig.canvas.draw()
            # acc_display.update(obj=acc_fig)

            proj_directory = "/Users/mkshah605/Documents/GitHub/RC/NMT_IndicLang/"

            loss_fig.savefig(proj_directory + "images/loss_fig.png")
            acc_fig.savefig(proj_directory + "images/acc_fig.png")
        
        return loss_logs, acc_logs




        # for epoch in range(1, n_epochs + 1):
        #     loss = self.train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        #     print_loss_total += loss
        #     plot_loss_total += loss

        #     if epoch % print_every == 0:
        #         print_loss_avg = print_loss_total / print_every
        #         print_loss_total = 0
        #         print('%s (%d %d%%) %.4f' % (self.timeSince(start, epoch / n_epochs),
        #                                     epoch, epoch / n_epochs * 100, print_loss_avg))

        #     if epoch % plot_every == 0:
        #         plot_loss_avg = plot_loss_total / plot_every
        #         plot_losses.append(plot_loss_avg)
        #         plot_loss_total = 0

        # self.showPlot(plot_losses)














