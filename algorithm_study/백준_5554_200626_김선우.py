import sys
input = sys.stdin.readline

h_to_s = int(input())
s_to_p = int(input())
p_to_a = int(input())
a_to_h = int(input())

tot_min = (h_to_s + s_to_p + p_to_a + a_to_h) // 60
tot_sec = (h_to_s + s_to_p + p_to_a + a_to_h) % 60
print(tot_min)
print(tot_sec)