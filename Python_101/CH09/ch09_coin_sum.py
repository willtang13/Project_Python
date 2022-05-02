def coin_sum_r(total, coins):
    if (len(coins)) == 1:
        return 1
    elif (total < coins[-1]):
        return coin_sum_r(total, coins[:-1])
    else:
        return (coin_sum_r(total-coins[-1], coins) + 
                coin_sum_r(total, coins[:-1]))
    

coins_england = (1, 2, 5, 10, 20, 50, 100, 200)


def coin_sum_t(total, coins): #動態規劃法/建表法
    ways = [1] + ([0] * total)
    for coin in coins:
        for i in range(coin, total+1):
            ways[i] += ways[i-coin]
    return ways[total]

print(coin_sum_r(200, coins_england))

print(coin_sum_t(200, coins_england))