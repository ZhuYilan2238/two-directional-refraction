"""General-purpose training script for image-to-image translation."""
import time
import copy
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   
    dataset = create_dataset(opt) 
    dataset_size = len(dataset)    
    print('The number of training images = %d' % dataset_size)

    if not opt.no_val:
        val_opt = copy.deepcopy(opt)
        val_opt.phase = 'val'
        val_opt.serial_batches = True
        val_opt.no_flip = True

        val_dataset = create_dataset(val_opt)
        val_dataset_size = len(val_dataset)
        print('The number of validation images = %d' % val_dataset_size)

    model = create_model(opt)      
    model.setup(opt)               
    visualizer = Visualizer(opt)   
    total_iters = 0                

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0                  
        visualizer.reset()            
        model.update_learning_rate()    
        train_loss_sums = {name: 0.0 for name in model.loss_names}
        train_samples = 0
        for i, data in enumerate(dataset): 
            iter_start_time = time.time()  
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         
            model.optimize_parameters()   
            
            train_losses = model.get_current_losses()
            batch_size = model.real_A.size(0)
            for name in train_loss_sums:
                train_loss_sums[name] += train_losses[name] * batch_size            
            train_samples += batch_size
    
            if total_iters % opt.display_freq == 0:   
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        for name in train_loss_sums:
            train_loss_sums[name] /= train_samples

        print('Training epoch %d:' % epoch, train_loss_sums)   
        
        if not opt.no_val:
            model.eval()
            val_loss_sums = {name: 0.0 for name in model.loss_names}
            val_samples = 0
            for val_data in val_dataset:
                model.set_input(val_data)
                model.val_loss()
                losses = model.get_current_losses()
                batch_size = model.real_A.size(0)
                for name in val_loss_sums:
                    val_loss_sums[name] += losses[name] * batch_size
                val_samples += batch_size

            for name in val_loss_sums:
                val_loss_sums[name] /= val_samples
            visualizer.print_val_losses(epoch, val_loss_sums)
            model.netG.train()
            model.netD.train()
  
        if epoch % opt.save_epoch_freq == 0:             
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
