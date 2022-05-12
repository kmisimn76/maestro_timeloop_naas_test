        #hw_gen_bar = tqdm(total=num_gen, desc="GA search for HW", position=0)
        #hw_gen_log = tqdm(total=0, position=1, bar_format='{desc}')
        #map_gen_bar = tqdm(total=num_gen, position=2)
        #map_gen_log = tqdm(total=0, position=3, bar_format='{desc}')

        for hw_generation in range(num_gen):
            best_gen_reward =None
            parents = HWGene.select_parents(hw_new_population, hw_fitness
                                            num_parents)

            offspring_crossover = HWGene.crossover(parents,
                                            num_pop-num_parents)
            offspring_mutation = HWGene.mutation(offspring_crossover)

            hw_new_population[0:parents.shape[0], :] = parents
            hw_new_population[parents.shape[0]:, :] = offspring_mutation
            num_invalid_node = 0

            for hw_pop_i in range(num_pop):
                hw_gene = hw_new_population[hw_pop_i] #Select HW Gene

                #Optimize Mapping Gene
                tot_hw_reward = 0.0
                total_used_constraint = 0.0
                for lr_i in range(num_layers):
                    map_iter = 0
                    #map_gen_bar.set_description_str(desc="GA search for mapping layer {}".format(lr_i))
                    self.map_best_reward[lr_i] = float("-Inf")
                    for map_generation in range(num_gen):
                        best_gen_map_reward = None
                        parents = MapGene.select_parents(map_new_population[lr_i], map_fitness[lr_i],
                                                        num_parents)
                        offspring_crossover = MapGene.crossover(parents,
                                                        num_pop-num_parents)
                        offspring_mutation = MapGene.mutation(offspring_crossover)

                        map_new_population[lr_i][0:parents.shape[0], :] = parents
                        map_new_population[lr_i][parents.shape[0]:, :] = offspring_mutation
                        map_num_invalid_node = 0
                        for map_pop_i in range(num_pop):
                            map_gene = map_new_population[lr_i][map_pop_i]
                            reward, used_constraint = self.exterior_search(self.model_defs[lr_i], hw_gene, map_gene)
                            if reward is None: # invalid mapping
                                reward = float("-Inf")
                                #print("Error with reward")
                                #print(new_population[i])
                                #exit(-1)
                                map_num_invalid_node += 1
                            elif used_constraint > self.constraint_value: # not met constraints
                                reward = float("-Inf")
                                map_num_invalid_node += 1
                            if reward > self.map_best_reward[lr_i]:
                                best_gen_map_reward = reward
                                self.map_best_reward[lr_i] = reward
                                self.map_best_sol[lr_i] = map_gene
                            map_fitness[lr_i][map_pop_i] = reward
                        self.map_best_rewards_iteration[lr_i].append(map_iter)
                        self.map_best_rewards[lr_i].append(self.map_best_reward[lr_i])
                        map_iter += 1
                        #map_gen_bar.update(1)
                        if best_gen_map_reward  is not None:
                            #self.fd.write("\nMap Generation {}: new best award reward: {:9e}".format(map_generation+1, self.map_best_reward[lr_i])) if self.fd else None
                            print("\nMap Generation {} for HWgen{} Layer{}: new best award reward: {:9e}".format(map_generation+1, hw_generation, lr_i, self.map_best_reward[lr_i]))
                            #map_gen_log.set_description_str("Map Generation {}: new best award reward: {:9e}".format(map_generation+1, self.map_best_reward[lr_i]))
                    # == end of Mapping GA each layer
                    reward, used_constraint = self.exterior_search(self.model_defs[lr_i], hw_gene, self.map_best_sol[lr_i]) # get reward of best mapping
                    #print("Observation ", self.observation)
                    #print(reward, used_constraint)
                    if reward == None:
                        tot_hw_reward=None
                        total_used_constraint = None
                        print("None")
                        break
                    tot_hw_reward += reward
                    total_used_constraint += used_constraint
                    #map_gen_bar.update(-num_gen)
                # == end of Mapping GA all layer
                reward = tot_hw_reward
                if reward is None:
                    reward = float("-Inf")
                    #print("Error with reward")
                    #print(new_population[i])
                    #exit(-1)
                elif total_used_constraint//num_layers > self.constraint_value:
                    reward = float("-Inf")
                    num_invalid_node += 1
                if reward > self.best_reward:
                    best_gen_reward = reward
                    self.best_reward = reward
                    self.best_reward_constraint = total_used_constraint/num_layers
                    self.best_sol = hw_gene

                hw_fitness[hw_pop_i] = reward
                iteration += 1
                self.best_rewards_iteration.append(iteration)
                self.best_rewards_constraint.append(self.best_reward_constraint)
                self.best_rewards.append(self.best_reward)
            if best_gen_reward  is not None:
                #self.fd.write("\nHW Generation {}: new best award reward: {:9e}".format(hw_generation+1, self.best_reward)) if self.fd else None
                print("\nHW Generation {}: new best award reward: {:9e}".format(hw_generation+1, self.best_reward))
                #hw_gen_log.set_description_str("HW Generation {}: new best award reward: {:9e}".format(hw_generation+1, self.best_reward))
            self.count_invalid += num_invalid_node
            self.save_chkpt()
            #hw_gen_bar.update(1)
        # == end of HW GA

